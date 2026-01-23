# train_ssl_baseline.py

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

from ctw1500_subset import CTW1500Subset, ctw1500_collate_fn


# ----------------------------------
# SemiETS hard threshold
# ----------------------------------
T_D = 0.5


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
        target_lengths = batch["target_lengths"].to(device)

        image_ids = batch.get("image_ids", None)
        gt_texts = batch.get("gt_texts", None)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            # ----------------------------------
            # Student forward
            # ----------------------------------
            student_log_probs = ssl_model.student_forward(images)  # (B, T, V)
            student_log_probs = student_log_probs.permute(1, 0, 2)  # (T, B, V)

            # ----------------------------------
            # CTC-safe input lengths
            # ----------------------------------
            T = student_log_probs.size(0)
            B = student_log_probs.size(1)
            input_lengths = torch.full(
                size=(B,),
                fill_value=T,
                dtype=torch.long,
                device=student_log_probs.device
            )

            # ----------------------------------
            # HARD binary acceptance (baseline)
            # ----------------------------------
            baseline_weights = (det_conf > T_D).float()

            # ----------------------------------
            # Weighted CTC loss
            # ----------------------------------
            loss = ssl_model.criterion(
                student_log_probs,
                targets,
                input_lengths,
                target_lengths,
                baseline_weights
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # EMA update (kept for fair comparison)
        ssl_model.update_teacher()

        total_loss += loss.item()

        # ----------------------------------
        # Logging (failure analysis)
        # ----------------------------------
        det_conf_cpu = det_conf.detach().cpu()
        weights_cpu = baseline_weights.detach().cpu()

        for i in range(B):
            log_entry = {
                "epoch": epoch,
                "step": step,
                "image_id": image_ids[i] if image_ids else f"idx_{step}_{i}",
                "det_conf": float(det_conf_cpu[i]),
                "baseline_accept": bool(weights_cpu[i]),
                "loss": float(loss.item()) if weights_cpu[i] > 0 else 0.0
            }

            if gt_texts is not None:
                log_entry["gt_text"] = gt_texts[i]
                log_entry["word_length"] = len(gt_texts[i])

            experiment_logs.append(log_entry)

        if step % 10 == 0:
            print(
                f"[Step {step}] "
                f"Loss: {loss.item():.4f} | "
                f"Accepted samples: {weights_cpu.sum().item()}/{B}"
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

    ssl_model.student_recognizer.rnn.flatten_parameters()
    ssl_model.teacher_recognizer.rnn.flatten_parameters()

    # ----------------------------------
    # Optimizer (student only)
    # ----------------------------------
    optimizer = Adam(
        list(ssl_model.student_backbone.parameters()) +
        list(ssl_model.student_recognizer.parameters()),
        lr=1e-4
    )

    scaler = GradScaler()

    # ----------------------------------
    # Dataset
    # ----------------------------------
    vocab = "0123456789abcdefghijklmnopqrstuvwxyz"

    dataset = CTW1500Subset(
    image_dir="C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\SemiETS-Improvements\\data\\ctw\\images",
    annotation_dir="C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\SemiETS-Improvements\\data\\ctw\\ctw1500_train_labels",
    vocab=vocab,
    max_samples=1000
)

    dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=ctw1500_collate_fn,
    pin_memory=True
)

    # ----------------------------------
    # Training loop
    # ----------------------------------
    os.makedirs("experiment_logs_baseline", exist_ok=True)

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

        with open(f"experiment_logs_baseline/epoch_{epoch}.json", "w") as f:
            json.dump(experiment_logs, f, indent=2)

        print(f"[Epoch {epoch}] Avg loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
