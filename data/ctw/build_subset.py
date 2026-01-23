import os
import json
import xml.etree.ElementTree as ET


def parse_ctw_xml(xml_path):
    """
    Returns list of valid text strings from one CTW1500 XML
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    texts = []

    for textline in root.findall("image/Text"):
        transcription = textline.attrib.get("Transcription", "")
        transcription = transcription.strip()

        if transcription and transcription != "###":
            texts.append(transcription.lower())

    return texts


def build_ctw1500_subset(image_dir, xml_dir, output_json):
    samples = []

    for fname in sorted(os.listdir(xml_dir)):
        if not fname.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, fname)
        img_name = fname.replace(".xml", ".jpg")
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            continue

        texts = parse_ctw_xml(xml_path)

        for t in texts:
            samples.append({
                "image_path": img_path,
                "text": t
            })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"[CTW1500] Saved {len(samples)} samples → {output_json}")


if __name__ == "__main__":
    build_ctw1500_subset(
        image_dir="C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\SemiETS-Improvements\\data\\ctw\\images",
        xml_dir="C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\SemiETS-Improvements\\data\\ctw\\ctw1500_train_labels",
        output_json="C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\SemiETS-Improvements\\data\\ctw\\ctw1500_subset.json"
    )
