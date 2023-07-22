
import torch
import clip
from PIL import Image
import os
import torch.nn as nn
import torch
import clip
from PIL import Image
import os
import torch.nn as nn

class Clip_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_dir = os.path.join(os.getcwd(), "EditLeRF", "Clip_encoder", "modelckpt")
        self.clip_path = os.path.join(self.clip_dir, "clip_model_state_dict.pt")
        self.model = None  # Initialize self.model as None
        self.preprocess = None  # Initialize self.preprocess as None
        self.load_clip_model()  # Load the model and preprocess
        self.model.eval()  # Set the model to evaluation mode

    def get_clip_features(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features

    def get_clip_text_features(self, text):
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features

    def load_clip_model(self):
        if os.path.exists(self.clip_path):
            # Load the model's state dictionary from the local file
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.load_state_dict(torch.load(self.clip_path, map_location=self.device))
            print("clip already downloaded,loaded from local file")
        else:
            print("Downloading clip")

            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            # Save the model's state dictionary to a local file
            os.makedirs(self.clip_dir, exist_ok=True)
            torch.save(self.model.state_dict(), self.clip_path)


    class AppearenceMapper(nn.Module):
        def __init__(self, appearance_dim: int = 27):
            # define 2 layer mlp with channels 256 128 27 with relu activation functions
            # in the original paper this is 128 256 128
            super().__init__()
            self.lin1 = nn.Linear(256, 128)
            self.relu1 = nn.ReLU()
            self.lin2 = nn.Linear(128, appearance_dim)
            self.relu2 = nn.ReLU()
            self.mlp = nn.Sequential(self.lin1, self.relu1, self.lin2, self.relu2)
            # this network will use adam optimizer with lr e^-4 and decayed by 0.5 every 50k steps
        def forward(self, x):
            x = self.mlp(x)
            return x

def main():
    #simple demo to test clip encoder 
    # create clip encoder instance and get text features
    clip_encoder = Clip_encoder()
    text_features = clip_encoder.get_clip_text_features(['wall', 'chair', 'floor', 'table', 'door', 'couch', 'green','red','yellow'])
    print("Text features:", text_features)

if __name__ == "__main__":
    main()