"""
Logit-based Boundary Distillation using Prewitt Operator.
Removes morphological masking and operates directly on output probabilities.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast # Modern API

from .base import DistillationMethod


class BoundaryLogitsPrewitt(DistillationMethod, nn.Module):
    """
    Computes the boundary difference between Student and Teacher LOGITS 
    using the Prewitt operator.
    """

    def __init__(self, use_softmax=True, **config):
        # Initialize both parents correctly
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)
        self.use_softmax = use_softmax
        self._init_prewitt_kernels()

    def _init_prewitt_kernels(self) -> None:
        """
        Define 3D Prewitt kernels.
        """
        # Base 1D kernels
        diff = torch.tensor([-1, 0, 1], dtype=torch.float32)
        smooth = torch.tensor([1, 1, 1], dtype=torch.float32)

        # Construct 3D kernels using outer products
        # Z-direction edge
        k_z = torch.einsum('i,j,k->ijk', diff, smooth, smooth) 
        # Y-direction edge
        k_y = torch.einsum('i,j,k->ijk', smooth, diff, smooth)
        # X-direction edge
        k_x = torch.einsum('i,j,k->ijk', smooth, smooth, diff)

        # Reshape for F.conv3d: (Out=1, In=1, D, H, W)
        self.register_buffer("prewitt_z", k_z.view(1, 1, 3, 3, 3))
        self.register_buffer("prewitt_y", k_y.view(1, 1, 3, 3, 3))
        self.register_buffer("prewitt_x", k_x.view(1, 1, 3, 3, 3))

    def extract_edges(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Prewitt operator to extract edges magnitude.
        """
        # Disable Autocast to force FP32 math (Prewitt can be unstable in FP16)
        with autocast(device_type='cuda', enabled=False):
            x = x.float()
            
            B, C, D, H, W = x.shape
            x_flat = x.flatten(0, 1).unsqueeze(1)
            
            # Ensure kernels are on the same device as input
            k_z = self.prewitt_z.to(x.device)
            k_y = self.prewitt_y.to(x.device)
            k_x = self.prewitt_x.to(x.device)
            
            # Apply kernels
            edge_z = F.conv3d(x_flat, k_z, padding=1)
            edge_y = F.conv3d(x_flat, k_y, padding=1)
            edge_x = F.conv3d(x_flat, k_x, padding=1)
            
            magnitude = torch.sqrt(edge_z**2 + edge_y**2 + edge_x**2 + 1e-8)
            return magnitude.view(B, C, D, H, W)

    def forward(
        self, 
        student_features, 
        teacher_features, 
        target, 
        student_output=None, 
        teacher_output=None, 
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        if student_output is None or teacher_output is None:
            return torch.tensor(0.0, device=target.device), {}

        # 1. Align shapes
        if student_output.shape != teacher_output.shape:
             student_output = F.interpolate(
                 student_output, 
                 size=teacher_output.shape[2:], 
                 mode='trilinear', 
                 align_corners=False
             )

        # 2. Convert to Probabilities
        if self.use_softmax:
            # --- FIX: Auto-detect BraTS (Multi-label) vs Standard (Multi-class) ---
            if target.shape[1] > 1:
                # BraTS Case: Use Sigmoid so channels don't fight each other
                s_input = torch.sigmoid(student_output)
                t_input = torch.sigmoid(teacher_output)
            else:
                # Standard Case: Use Softmax
                s_input = F.softmax(student_output, dim=1)
                t_input = F.softmax(teacher_output, dim=1)
        else:
            s_input = student_output
            t_input = teacher_output

        # 3. Extract Edges
        with torch.no_grad():
            t_edges = self.extract_edges(t_input)
        
        s_edges = self.extract_edges(s_input)

        # 4. Point-wise Dot Product 
        # Weighted edge loss: Focuses edge matching on areas where probabilities are high
        s_weighted = s_input * s_edges
        t_weighted = t_input * t_edges

        # 5. Calculate Loss
        loss = F.mse_loss(s_weighted, t_weighted)

        return loss, {"prewitt_loss": loss.item()}

    def get_required_features(self) -> Dict[str, str]:
        """
        Required implementation of abstract method.
        Since we use logits, we need no intermediate features.
        """
        return {}