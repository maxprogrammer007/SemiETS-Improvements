import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import json
import os

from models.backbone import Backbone
from models.recognizer import CRNN
from models.teacher_student import TeacherStudentSSL
from losses.weighted_loss import WeightedCTCLoss

from data.ic15_subset import IC15Subset, ic15_collate_fn


# ----------------------------------
# Baseline simulation threshold
# ----------------------------------
EPS = 0.01   # samples below this are treated as "rejected" in baseline


def train_one_epoch(
    ssl_model,
    dataloader,
    optimizer,
    device,
    scaler,
    epoch,
    experiment_logs
):
    ssl_model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        det_conf = batch["det_conf"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        # Optional metadata (if available)
        image_ids = batch.get("image_ids", None)
        gt_texts = batch.get("gt_texts", None)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            loss, weights = ssl_model(
                images,
                det_conf,
                targets,
                input_lengths,
                target_lengths
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # EMA update (teacher)
        ssl_model.update_teacher()

        total_loss += loss.item()

        # ----------------------------------
        # Experiment logging (failure analysis)
        # ----------------------------------
        weights_cpu = weights.detach().cpu()
        det_conf_cpu = det_conf.detach().cpu()

        for i in range(len(weights_cpu)):
            log_entry = {
                "epoch": epoch,
                "step": step,
                "image_id": image_ids[i] if image_ids is not None else f"idx_{step}_{i}",
                "det_conf": float(det_conf_cpu[i]),
                "final_weight": float(weights_cpu[i]),
                "baseline_accept": bool(weights_cpu[i] > EPS),
                "loss": float(loss.item())
            }

            if gt_texts is not None:
                log_entry["gt_text"] = gt_texts[i]
                log_entry["word_length"] = len(gt_texts[i])

            experiment_logs.append(log_entry)

        if step % 10 == 0:
            print(
                f"[Step {step}] "
                f"Loss: {loss.item():.4f} | "
                f"Mean reliability: {weights.mean().item():.4f}"
            )

    return total_loss / max(len(dataloader), 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------
    # Build model
    # ----------------------------------
    backbone = Backbone().to(device)
    recognizer = CRNN().to(device)
    criterion = WeightedCTCLoss()

    ssl_model = TeacherStudentSSL(
        backbone=backbone,
        recognizer=recognizer,
        criterion=criterion,
        ema_decay=0.999
    ).to(device)

    # LSTM optimization (recommended)
    ssl_model.student_recognizer.rnn.flatten_parameters()
    ssl_model.teacher_recognizer.rnn.flatten_parameters()

    # ----------------------------------
    # Optimizer (STUDENT ONLY)
    # ----------------------------------
    optimizer = Adam(
        list(ssl_model.student_backbone.parameters()) +
        list(ssl_model.student_recognizer.parameters()),
        lr=1e-4
    )

    scaler = GradScaler()

    # ----------------------------------
    # Dataset & Dataloader
    # ----------------------------------
    vocab = "0123456789abcdefghijklmnopqrstuvwxyz"

    dataset = IC15Subset(
        image_dir="D:\\semiETS stuffs\\semiets_scratch\\data\\ic15\\images",
        annotation_json="D:\\semiETS stuffs\\semiets_scratch\\data\\ic15\\ic15_subset.json",
        vocab=vocab,
        max_samples=100,
        train=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,          # safe for 4GB VRAM
        shuffle=True,
        num_workers=2,
        collate_fn=ic15_collate_fn,
        pin_memory=True
    )

    # ----------------------------------
    # Training loop
    # ----------------------------------
    os.makedirs("experiment_logs", exist_ok=True)

    epochs = 5
    for epoch in range(epochs):
        experiment_logs = []

        avg_loss = train_one_epoch(
            ssl_model,
            dataloader,
            optimizer,
            device,
            scaler,
            epoch,
            experiment_logs
        )

        # Save logs for this epoch
        with open(f"experiment_logs/epoch_{epoch}.json", "w") as f:
            json.dump(experiment_logs, f, indent=2)

        print(f"[Epoch {epoch}] Avg loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
