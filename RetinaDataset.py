import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class RetinaDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.retina_frame = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        # Duplicate each row, one for the left image and one for the right image
        self.retina_frame = pd.concat([self.retina_frame]*2, ignore_index=True)
        self.eye_side = ['left'] * (len(self.retina_frame) // 2) + ['right'] * (len(self.retina_frame) // 2)  

    def __len__(self):
        return len(self.retina_frame)

    def __getitem__(self, idx):
        # Construct the image file path
        img_name = os.path.join(self.img_dir, f"{self.retina_frame.iloc[idx, 0]}_{self.eye_side[idx]}.jpg")

        # Load the image directly as a PIL Image
        image = Image.open(img_name).convert('RGB')

        # Rerieve the age from the dataframe
        age = torch.tensor(self.retina_frame.iloc[idx, 1], dtype=torch.float32)

        return {'image': image, 'age': age}
