import os
import json

gt_dir = "data/ic15/gt"
img_dir = "data/ic15/images"

samples = []

for gt_file in os.listdir(gt_dir):
    if not gt_file.endswith(".txt"):
        continue

    img_name = gt_file.replace("gt_", "").replace(".txt", ".jpg")
    img_path = os.path.join(img_dir, img_name)

    if not os.path.exists(img_path):
        continue

    with open(os.path.join(gt_dir, gt_file), "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip().split(",")[-1]
            if text == "###":
                continue

            samples.append({
                "image": img_name,
                "text": text.lower()
            })

# Keep it SMALL for now
samples = samples[:100]

with open("data/ic15/ic15_subset.json", "w") as f:
    json.dump(samples, f, indent=2)

print("Saved", len(samples), "samples")
