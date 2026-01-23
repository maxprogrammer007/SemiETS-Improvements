# data/ctw1500_subset.py

import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import resize_and_normalize
from utils.text import text_to_indices


class CTW1500Subset(Dataset):
    """
    CTW1500 XML-based dataset loader (SemiETS compatible)

    One text instance = one sample
    """

    def __init__(
        self,
        image_dir,
        annotation_dir,
        vocab,
        max_samples=None,
        train=True
    ):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.vocab = vocab
        self.train = train

        self.samples = []
        self._build_index()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"[CTW1500] Loaded {len(self.samples)} samples")

        if len(self.samples) == 0:
            raise RuntimeError(
                "CTW1500Subset: 0 samples loaded. "
                "Check XML annotations or image paths."
            )

    # --------------------------------------------------
    def _build_index(self):
        xml_files = sorted(f for f in os.listdir(self.annotation_dir) if f.endswith(".xml"))

        for xml_name in xml_files:
            xml_path = os.path.join(self.annotation_dir, xml_name)
            img_name = xml_name.replace(".xml", ".jpg")
            img_path = os.path.join(self.image_dir, img_name)

            if not os.path.exists(img_path):
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # CTW1500 XML structure: <Annotataions><image><box><label>
            for image_elem in root.findall("image"):
                for box in image_elem.findall("box"):
                    label_node = box.find("label")

                    if label_node is None or label_node.text is None:
                        continue

                    text = label_node.text.strip()
                    if text == "" or text == "###":
                        continue

                    self.samples.append({
                        "image_path": img_path,
                        "text": text.lower()
                    })

    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = sample["image_path"]
        text = sample["text"]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_and_normalize(image)

        targets = text_to_indices(text, self.vocab)

        # Simulated detection confidence (SemiETS-style)
        det_conf = np.random.uniform(0.3, 1.0)

        return {
            "images": torch.from_numpy(image).float(),
            "det_conf": torch.tensor(det_conf, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.long),
            "target_lengths": torch.tensor(len(targets), dtype=torch.long),
            "image_ids": os.path.basename(img_path),
            "gt_texts": text
        }


# --------------------------------------------------
# Collate function (CTC-safe)
# --------------------------------------------------
def ctw1500_collate_fn(batch):
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
