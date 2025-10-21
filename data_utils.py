import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TrafficDataset(Dataset):
    def __init__(self, root_dir, split="train", instance_prompt="a photo of sks_vehicle", limit=None):
        self.root_dir = os.path.join(root_dir, split)
        self.instance_prompt = instance_prompt

        self.image_paths = []
        self.labels = []

        for class_name in sorted(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(class_name)

        if limit:
            self.image_paths = self.image_paths[:limit]
            self.labels = self.labels[:limit]

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # normalize to [-1, 1]
        ])

        print(f"[Dataset] Loaded {len(self.image_paths)} {split} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        prompt = f"{self.instance_prompt} {label}"

        return {"instance_images": image, "instance_prompt": prompt}


def make_dreambooth_dataset(num_samples=50, split="train"):

    drive_path = "trafic_data"

    dataset = TrafficDataset(root_dir=drive_path, split=split, limit=num_samples)
    return dataset
