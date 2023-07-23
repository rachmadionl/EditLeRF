import torch
import clip
import numpy as np

from PIL import Image
from torchmetrics.multimodal import CLIPScore


class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        #self.upsample = torch.nn.Upsample(scale_factor=7)
        #self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text):
        # image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        image = self.preprocess(image).unsqueeze(0).to('cuda')
        #image = self.avg_pool(self.upsample(image))
        l = self.model(image, text)[0]
        print(l)
        similarity = 1 - l / 100
        return similarity#.mean()


clip_loss = CLIPLoss()

text = 'a cat'
text_inputs = torch.cat([clip.tokenize(text)]).to('cuda')
image = Image.open("cat.jpg")

loss = clip_loss(image, text_inputs)
print(loss)


def get_clip_score(image_path, text):
    # Load the pre-trained CLIP model and the image
    model, preprocess = clip.load('ViT-B/32')
    image = Image.open(image_path)

    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)
    image_input = torch.rand(3, 4960, 1)
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score

image_path = "cat.jpg"
text = "This is a cat"

score = get_clip_score(image_path, text)
score = 1. - score
print(f"CLIP Score: {score}")

image = Image.open(image_path)
_, preprocess = clip.load('ViT-B/32')
image_input = torch.from_numpy(np.array(image)).unsqueeze(0)

metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
score = metric(image_input, text)
score = 1. - score / 100
print(f'CLIP torchmetrics score: {score}')
