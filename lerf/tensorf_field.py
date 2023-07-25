# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TensoRF Field"""


from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import Encoding, Identity, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field


class TensoRFField(Field):
    """TensoRF Field"""

    def __init__(
        self,
        aabb: Tensor,
        # the aabb bounding box of the dataset
        feature_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for appearance encoding outputs
        direction_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for ray direction
        density_encoding: Encoding = Identity(in_dim=3),
        # the tensor encoding method used for scene density
        color_encoding: Encoding = Identity(in_dim=3),
        # the tensor encoding method used for scene color
        appearance_dim: int = 27,
        # the number of dimensions for the appearance embedding
        head_mlp_num_layers: int = 2,
        # number of layers for the MLP
        head_mlp_layer_width: int = 128,
        # layer width for the MLP
        use_sh: bool = True,
        # whether to use spherical harmonics as the feature decoding function
        sh_levels: int = 2,
        # number of levels to use for spherical harmonics
        train_clip: bool = False,
        # whether to train color stylization
        text_features: Optional[torch.Tensor] = None,
        # whether to use appearance mapper
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.feature_encoding = feature_encoding
        self.direction_encoding = direction_encoding
        self.density_encoding = density_encoding
        self.color_encoding = color_encoding
        self.train_clip = train_clip
        self.text_features = text_features

        self.mlp_head = MLP(
            in_dim=appearance_dim + 3 + self.direction_encoding.get_out_dim() + self.feature_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )

        self.use_sh = use_sh
        print(f"IS SH USED? {self.use_sh}")

        if self.use_sh:
            self.sh = SHEncoding(sh_levels)
            self.B = nn.Linear(
                in_features=self.color_encoding.get_out_dim(), out_features=3 * self.sh.get_out_dim(), bias=False
            )
        else:
            self.B = nn.Linear(in_features=self.color_encoding.get_out_dim(), out_features=appearance_dim, bias=False)

        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(), activation=nn.Sigmoid())

        if self.train_clip:
            if self.text_features is not None:
                self.appearance_mapper = MLP(
                    in_dim=512,
                    num_layers=2,
                    layer_width=128,
                    out_dim=appearance_dim,
                    activation=nn.ReLU(),
                    out_activation=nn.ReLU(),
                )
            # self.mlp_head = self.mlp_head.eval()
            # self.field_output_rgb = self.field_output_rgb.eval()
            self.B = self.B.eval()
            self.color_encoding = self.color_encoding.eval()
            self.feature_encoding = self.feature_encoding.eval()
            self.direction_encoding = self.direction_encoding.eval()
            self.density_encoding = self.density_encoding.eval()

            for param in self.B.parameters():
                param.requires_grad = False
            for param in self.color_encoding.parameters():
                param.requires_grad = False
            for param in self.feature_encoding.parameters():
                param.requires_grad = False
            for param in self.direction_encoding.parameters():
                param.requires_grad = False
            for param in self.density_encoding.parameters():
                param.requires_grad = False

    def get_density(self, ray_samples: RaySamples) -> Tensor:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        density = self.density_encoding(positions)
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        relu = torch.nn.ReLU()
        density_enc = relu(density_enc)
        return density_enc

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        rgb_features = self.color_encoding(positions)
        rgb_features = self.B(rgb_features)

        if self.text_features is not None:
            import pdb; pdb.set_trace()
            color_features = self.appearance_mapper(self.text_features)
            rgb_features = rgb_features + color_features

        if self.use_sh:
            sh_mult = self.sh(d)[:, :, None]
            rgb_sh = rgb_features.view(sh_mult.shape[0], sh_mult.shape[1], 3, sh_mult.shape[-1])
            rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
        else:
            d_encoded = self.direction_encoding(d)
            rgb_features_encoded = self.feature_encoding(rgb_features)

            out = self.mlp_head(torch.cat([rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))  # type: ignore
            rgb = self.field_output_rgb(out)

        return rgb

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[Tensor] = None,
        bg_color: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals is True:
            raise ValueError("Surface normals are not currently supported with TensoRF")
        if mask is not None and bg_color is not None:
            base_density = torch.zeros(ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(ray_samples[:, :, None].shape)
            if mask.any():
                input_rays = ray_samples[mask, :]
                if self.train_clip:
                    with torch.no_grad():
                        density = self.get_density(input_rays)
                    rgb = self.get_outputs(input_rays, None).type(torch.float32)
                else:
                    density = self.get_density(input_rays)
                    rgb = self.get_outputs(input_rays, None).type(torch.float32)

                base_density[mask] = density
                base_rgb[mask] = rgb

                if self.train_clip:
                    base_rgb.requires_grad_()
                else:
                    base_rgb.requires_grad_()
                    base_density.requires_grad_()

            density = base_density
            rgb = base_rgb
        else:
            if self.train_clip:
                with torch.no_grad():
                    density = self.get_density(ray_samples)
                rgb = self.get_outputs(ray_samples, None)
            else:
                density = self.get_density(ray_samples)
                rgb = self.get_outputs(ray_samples, None)

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}
