from typing import Optional

from torch import nn
import clip
import open_clip
import torch


class ClipEncoding(nn.Module):
    def __init__(self,
                 clip_model_type: str = 'ViT-B-32',
                 pretrained_type: str = 'laion2b_s34b_b79k',
                 device: Optional[str] = 'cuda'
                 ) -> None:
        super().__init__()
        self.device = device
        if '16' in clip_model_type:
            self.precision = 'fp16'
        else:
            self.precision = 'fp32'

        encoder, _, _ = open_clip.create_model_and_transforms(clip_model_type, pretrained=pretrained_type,
                                                              precision=self.precision)
        self.encoder = encoder.to(device)
        self.tokenizer = open_clip.get_tokenizer(clip_model_type)

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        text_features = self.encoder.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        image_features = self.encoder.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, rgb_outputs: torch.Tensor, text_features: torch.Tensor, ) -> torch.Tensor:
        image_features = self.encode_image(rgb_outputs)
        loss = 1 - image_features @ text_features.T
        return loss


class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")

    def forward(self, image, text):
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity
