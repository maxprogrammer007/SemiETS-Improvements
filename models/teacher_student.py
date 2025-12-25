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
        consistency_weight=0.1
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

        self.freeze_teacher()

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
            t_param.data = (
                self.ema_decay * t_param.data
                + (1.0 - self.ema_decay) * s_param.data
            )

        for t_param, s_param in zip(
            self.teacher_recognizer.parameters(),
            self.student_recognizer.parameters()
        ):
            t_param.data = (
                self.ema_decay * t_param.data
                + (1.0 - self.ema_decay) * s_param.data
            )

    @torch.no_grad()
    def teacher_forward(self, images):
        feats = self.teacher_backbone(images)
        log_probs = self.teacher_recognizer(feats)
        return log_probs

    # --------------------------------------------------
    # Student forward
    # --------------------------------------------------
    def student_forward(self, images):
        feats = self.student_backbone(images)
        log_probs = self.student_recognizer(feats)
        return log_probs

    # --------------------------------------------------
    # Main forward
    # --------------------------------------------------
    def forward(
        self,
        images,
        det_conf,
        targets,
        input_lengths,
        target_lengths
    ):
        """
        images: (B, 3, H, W)
        det_conf: (B,)
        """

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

            # Main teacher output (reference view)
            teacher_log_probs = teacher_outputs[0]

        # ======================================
        # Soft reliability (TD / TR / CC replacement)
        # ======================================
        det_w = detection_reliability(det_conf, self.det_thr, self.tau)
        rec_w = recognition_reliability(teacher_log_probs, self.alpha)
        final_w = combine_reliability(det_w, rec_w)

        # ======================================
        # Student prediction
        # ======================================
        student_log_probs = self.student_forward(images)
        student_log_probs = student_log_probs.permute(1, 0, 2)  # (T, B, V)

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
