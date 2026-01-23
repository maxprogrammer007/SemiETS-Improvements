import os
import json
import numpy as np
from scipy.io import loadmat


def extract_texts_from_mat(mat):
    """
    Extract word-level transcriptions from TotalText .mat file
    Compatible with object-array based gt format
    """
    valid_texts = []

    if "gt" not in mat:
        return valid_texts

    gt = mat["gt"]

    # gt shape: (N, 6), dtype=object
    for i in range(gt.shape[0]):
        t = gt[i, 4]  # TEXT FIELD

        # unwrap numpy arrays
        if isinstance(t, np.ndarray):
            t = t.item()

        # decode bytes
        if isinstance(t, bytes):
            t = t.decode("utf-8")

        t = str(t).strip()

        # ignore invalid regions
        if t == "#" or t == "###" or len(t) == 0:
            continue

        valid_texts.append(t.lower())

    return valid_texts


def build_totaltext_json(image_dir, annotation_dir, output_json):
    samples = []

    for fname in sorted(os.listdir(annotation_dir)):
        if not fname.endswith(".mat"):
            continue

        base = fname.replace("gt_", "").replace(".mat", "")

        img_path = os.path.join(image_dir, base + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, base + ".png")

        ann_path = os.path.join(annotation_dir, fname)

        if not os.path.exists(img_path):
            continue

        mat = loadmat(ann_path)
        texts = extract_texts_from_mat(mat)

        if len(texts) == 0:
            continue

        samples.append({
            "image_path": img_path,
            "texts": texts
        })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_json}")


if __name__ == "__main__":
    build_totaltext_json(
        image_dir=r"C:\Users\abhin\OneDrive\Documents\GitHub\SemiETS-Improvements\data\totaltext\images",
        annotation_dir=r"C:\Users\abhin\OneDrive\Documents\GitHub\SemiETS-Improvements\data\totaltext\annotations",
        output_json=r"C:\Users\abhin\OneDrive\Documents\GitHub\SemiETS-Improvements\data\totaltext\totaltext_subset.json"
    )
