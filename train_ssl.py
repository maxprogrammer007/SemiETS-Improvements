# train_ssl.py

import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from models.backbone import Backbone
from models.recognizer import CRNN
from models.teacher_student import TeacherStudentSSL
from losses.weighted_loss import WeightedCTCLoss

from data.ic15_subset import IC15Subset, ic15_collate_fn


def train_one_epoch(
    ssl_model,
    dataloader,
    optimizer,
    device,
    scaler
):
    ssl_model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        det_conf = batch["det_conf"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

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
        ema_decay=0.999,
        consistency_weight=0.1
    ).to(device)

    # LSTM optimization (optional but recommended)
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
    epochs = 5
    for epoch in range(epochs):
        avg_loss = train_one_epoch(
            ssl_model,
            dataloader,
            optimizer,
            device,
            scaler
        )
        print(f"[Epoch {epoch}] Avg loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
