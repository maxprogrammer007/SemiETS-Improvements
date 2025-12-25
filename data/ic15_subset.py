# data/ic15_subset.py

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

from data.transforms import build_transforms


class IC15Subset(Dataset):
    """
    Minimal IC15 dataset loader for SSL experiments.

    This loader focuses on:
    - image loading
    - text transcription
    - CTC-ready targets

    Detection confidence is simulated to study SSL behavior.
    """

    def __init__(
        self,
        image_dir,
        annotation_json,
        vocab,
        max_samples=200,
        img_size=(224, 224),
        train=True
    ):
        super().__init__()

        self.image_dir = image_dir
        self.vocab = vocab
        self.char2idx = {c: i + 1 for i, c in enumerate(vocab)}  # 0 reserved for blank
        self.max_samples = max_samples

        with open(annotation_json, "r") as f:
            self.annotations = json.load(f)

        self.samples = self.annotations[:max_samples]

        self.transform = build_transforms(img_size, train=train)

    def encode_text(self, text):
        """
        Convert string to CTC target indices
        """
        targets = []
        for c in text.lower():
            if c in self.char2idx:
                targets.append(self.char2idx[c])
        return torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        text = item["text"]
        targets = self.encode_text(text)

        # -----------------------------
        # Simulated detection confidence
        # -----------------------------
        det_conf = torch.rand(1).clamp(0.2, 0.9)

        sample = {
            "images": image,
            "det_conf": det_conf,
            "targets": targets,
            "input_lengths": torch.tensor([image.shape[-1] // 32]),
            "target_lengths": torch.tensor([len(targets)])
        }

        return sample
    
    
def ic15_collate_fn(batch):
        """
        Custom collate function for CTC-based recognition
        """
        images = torch.stack([b["images"] for b in batch])
        det_conf = torch.cat([b["det_conf"] for b in batch])

        targets = torch.cat([b["targets"] for b in batch])
        target_lengths = torch.cat([b["target_lengths"] for b in batch])
        input_lengths = torch.cat([b["input_lengths"] for b in batch])

        return {
            "images": images,
            "det_conf": det_conf,
            "targets": targets,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths
        }

