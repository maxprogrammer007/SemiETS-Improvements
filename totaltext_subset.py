import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from data.transforms import resize_and_normalize
from utils.text import text_to_indices


class TotalTextSubset(Dataset):
    def __init__(
        self,
        image_dir,
        annotation_json,
        vocab,
        max_samples=None,
        train=True
    ):
        self.image_dir = image_dir
        self.vocab = vocab
        self.train = train

        with open(annotation_json, "r") as f:
            self.samples = json.load(f)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = sample["image_path"]
        texts = sample["texts"]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_and_normalize(image)

        # Use first word (consistent with IC15Subset simplification)
        text = texts[0] if len(texts) > 0 else ""

        targets = text_to_indices(text, self.vocab)

        # Simulated detection confidence
        det_conf = np.random.uniform(0.3, 1.0)

        return {
            "images": torch.from_numpy(image).float(),
            "det_conf": torch.tensor(det_conf, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.long),
            "target_lengths": torch.tensor(len(targets), dtype=torch.long),
            
            "image_ids": img_path.split("/")[-1],
            "gt_texts": text
        }


def totaltext_collate_fn(batch):
    images = torch.stack([b["images"] for b in batch])
    det_conf = torch.stack([b["det_conf"] for b in batch])

    targets = torch.cat([b["targets"] for b in batch])
    target_lengths = torch.stack([b["target_lengths"] for b in batch])

    image_ids = [b["image_ids"] for b in batch]
    gt_texts = [b["gt_texts"] for b in batch]

    return {
        "images": images,
        "det_conf": det_conf,
        "targets": targets,
        "target_lengths": target_lengths,
        "image_ids": image_ids,
        "gt_texts": gt_texts
    }

