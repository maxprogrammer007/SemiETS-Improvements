# models/teacher_student.py

import torch
import torch.nn as nn
import copy

from utils.reliability import (
    detection_reliability,
    recognition_reliability,
    combine_reliability
)

from utils.consistency import multi_view_consistency
from utils.pseudo_label import perturb_images


class TeacherStudentSSL(nn.Module):
    def __init__(
        self,
        backbone,
        recognizer,
        criterion,
        det_thr=0.5,
        tau=0.1,
        alpha=1.0,
        ema_decay=0.999,
        consistency_weight=0.1,
        max_reliability=0.2,
        warmup_epochs=5
    ):
        super().__init__()

        # ============================
        # Student networks
        # ============================
        self.student_backbone = backbone
        self.student_recognizer = recognizer

        # ============================
        # Teacher networks (EMA copies)
        # ============================
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_recognizer = copy.deepcopy(recognizer)

        self.criterion = criterion

        # Hyperparameters
        self.det_thr = det_thr
        self.tau = tau
        self.alpha = alpha
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight

        # Reliability curriculum
        self.max_reliability = max_reliability
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0  # updated externally

        self.freeze_teacher()

    # --------------------------------------------------
    # Epoch setter (called from training loop)
    # --------------------------------------------------
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    # --------------------------------------------------
    # Teacher utilities
    # --------------------------------------------------
    def freeze_teacher(self):
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_recognizer.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """
        EMA update of teacher parameters
        """
        for t_param, s_param in zip(
            self.teacher_backbone.parameters(),
            self.student_backbone.parameters()
        ):
            t_param.data.mul_(self.ema_decay).add_(
                s_param.data, alpha=(1.0 - self.ema_decay)
            )

        for t_param, s_param in zip(
            self.teacher_recognizer.parameters(),
            self.student_recognizer.parameters()
        ):
            t_param.data.mul_(self.ema_decay).add_(
                s_param.data, alpha=(1.0 - self.ema_decay)
            )

    @torch.no_grad()
    def teacher_forward(self, images):
        feats = self.teacher_backbone(images)
        log_probs = self.teacher_recognizer(feats)
        return log_probs  # (B, T, V)

    # --------------------------------------------------
    # Student forward
    # --------------------------------------------------
    def student_forward(self, images):
        feats = self.student_backbone(images)
        log_probs = self.student_recognizer(feats)
        return log_probs  # (B, T, V)

    # --------------------------------------------------
    # Main forward
    # --------------------------------------------------
    def forward(
        self,
        images,
        det_conf,
        targets,
        target_lengths
    ):
        """
        images: (B, 3, H, W)
        det_conf: (B,)
        targets: (sum(target_lengths),)
        target_lengths: (B,)
        """

        B = images.size(0)

        # ======================================
        # Teacher multi-view predictions
        # ======================================
        with torch.no_grad():
            views = [
                images,
                perturb_images(images),
                perturb_images(images)
            ]

            teacher_outputs = [
                self.teacher_forward(v) for v in views
            ]

            teacher_log_probs = teacher_outputs[0]  # (B, T, V)

        # ======================================
        # Soft reliability (TD / TR / CC)
        # ======================================
        det_w = detection_reliability(det_conf, self.det_thr, self.tau)
        rec_w = recognition_reliability(teacher_log_probs, self.alpha)
        final_w = combine_reliability(det_w, rec_w)

        # ======================================
        # Reliability warm-up (soft curriculum)
        # ======================================
        ramp = min(
            (self.current_epoch + 1) / max(self.warmup_epochs, 1),
            1.0
        )
        max_w = self.max_reliability * ramp
        final_w = torch.clamp(final_w, max=max_w)

        # ======================================
        # Student prediction
        # ======================================
        student_log_probs = self.student_forward(images)  # (B, T, V)
        student_log_probs = student_log_probs.permute(1, 0, 2)  # (T, B, V)

        # ======================================
        # Dynamically compute input_lengths
        # ======================================
        T = student_log_probs.size(0)
        input_lengths = torch.full(
            size=(B,),
            fill_value=T,
            dtype=torch.long,
            device=student_log_probs.device
        )

        # ======================================
        # Weighted CTC loss
        # ======================================
        ssl_loss = self.criterion(
            student_log_probs,
            targets,
            input_lengths,
            target_lengths,
            final_w
        )

        # ======================================
        # Multi-view consistency loss
        # ======================================
        consistency_loss = multi_view_consistency(teacher_outputs)

        # ======================================
        # Total loss
        # ======================================
        total_loss = ssl_loss + self.consistency_weight * consistency_loss

        return total_loss, final_w
