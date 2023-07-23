from dataclasses import dataclass, field
from typing import Tuple, Type

from PIL import Image
import numpy as np
import torch
import torchvision

import open_clip
from lerf.encoders.image_encoder import (BaseImageEncoder,
                                         BaseImageEncoderConfig)

@dataclass
class OpenCLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")


class OpenCLIPNetwork(BaseImageEncoder):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        # model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        # self.positive_input = ViewerText("LERF Positives", "", cb_hook=self.gui_cb)

        # self.positives = self.positive_input.value.split(";")
        # self.negatives = self.config.negatives
        # with torch.no_grad():
        #     tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
        #     self.pos_embeds = model.encode_text(tok_phrases)
        #     tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
        #     self.neg_embeds = model.encode_text(tok_phrases)
        # self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        # self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        # assert (
        #     self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        # ), "Positive and negative embeddings must have the same dimensionality"
        # assert (
        #     self.pos_embeds.shape[1] == self.clip_n_dims
        # ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    # def gui_cb(self,element):
    #     self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = self.tokenizer([self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        return self.pos_embeds

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def encode_image(self, input):
        processed_input = self.process(input).half().unsqueeze(0)
        with torch.no_grad():
            enc_image = self.model.encode_image(processed_input)
        return processed_input, enc_image


# clip = OpenCLIPNetwork(OpenCLIPNetworkConfig)
# image_1 = np.array(Image.open('cat.jpg'))
# image_1 = torch.moveaxis(torch.from_numpy(image_1).type(torch.float).to('cuda'), 2, 0)
# image_1, enc_image = clip.encode_image(image_1)
# enc_image /= enc_image.norm(dim=-1, keepdim=True)

# text_input = 'a cat'
# enc_text = clip.set_positives(text_input)

# loss = torch.nn.CosineSimilarity()
# with torch.no_grad():
#     similarity = loss(enc_text, enc_image)

# print(similarity)

# 

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k',
                                                             precision='fp16')
model = model.to('cuda')
tokenizer = open_clip.get_tokenizer('ViT-B-16')

image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to('cuda').half()
text = tokenizer(["a diagram", "a dog", "a cat"]).to('cuda')

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T)
    print(text_probs)
    

class ClipEncoding:
    def __init__(self,
                 clip_model_type: str = 'ViT-B-16',
                 pretrained_type: str = 'laion2b_s34b_b88k',
                 device: str = 'cuda'
                 ) -> None:

        if '16' in clip_model_type:
            self.precision = 'fp16'
        else:
            self.precision = 'fp32'

        clip, _, _ = open_clip.create_model_and_transforms(clip_model_type, pretrained=pretrained_type,
                                                           precision=self.precision)
        self.clip = clip.to(device)
        self.tokenizer = open_clip.get_tokenizer(clip_model_type)

    def encode_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        if self.precision == 'fp16':
            image = image.half()
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def clip_loss(self, text_features: torch.Tensor, rgb_outputs: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image(rgb_outputs)
        return 1 - (image_features @ text_features.T)
