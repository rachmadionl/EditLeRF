from typing import Optional

import open_clip
import torch


class ClipEncoding:
    def __init__(self,
                 clip_model_type: str = 'ViT-B-16',
                 pretrained_type: str = 'laion2b_s34b_b88k',
                 device: Optional[str] = 'cuda'
                 ) -> None:
        self.device = device
        if '16' in clip_model_type:
            self.precision = 'fp16'
        else:
            self.precision = 'fp32'

        encoder, _, _ = open_clip.create_model_and_transforms(clip_model_type, pretrained=pretrained_type,
                                                              precision=self.precision)
        self.encoder = encoder.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(clip_model_type)

    def encode_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            text_features = self.encoder.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        # if self.precision == 'fp16':
        #     image = image.half()
        with torch.no_grad():
            image_features = self.encoder.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def clip_loss(self, text_features: torch.Tensor, rgb_outputs: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image(rgb_outputs)
        loss = 1 - image_features @ text_features.T
        return loss
