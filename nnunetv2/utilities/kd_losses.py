# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FitNet_Loss', 'KL_Loss', 'CWD_Loss', 'IFVD_Loss', 'SKD_Loss', 'DoubleSimKD_Loss', 'AT_Loss', 'PSD3D', 'CSD3D', 'CIRKD_Loss']


class FitNet_Loss(nn.Module):
    def __init__(self, s_channels: int, t_channels: int, is_3d: bool = False):
        """
        FitNet Loss with support for 2D or 3D inputs.

        Args:
            s_channels (int): Student feature channels.
            t_channels (int): Teacher feature channels.
            is_3d (bool): If True, use Conv3d; else Conv2d.
        """
        super(FitNet_Loss, self).__init__()
        Conv = nn.Conv3d if is_3d else nn.Conv2d
        self.regressor = Conv(s_channels, t_channels, kernel_size=1, bias=False)

    def forward(self, feat_S: torch.Tensor, feat_T: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 loss between projected student and teacher features.

        Args:
            feat_S (Tensor): Student features, shape [B, C, H, W] or [B, C, D, H, W]
            feat_T (Tensor): Teacher features, same shape as projected student features

        Returns:
            Tensor: Scalar MSE loss
        """
        feat_S_proj = self.regressor(feat_S)
        return F.mse_loss(feat_S_proj, feat_T)


class KL_Loss(nn.Module):
    '''
    Knowledge Distillation Loss using KL divergence (3D version)
    '''
    def __init__(self, temperature=3):
        super(KL_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, pred, soft):
        # pred and soft are tensors of shape (B, C, D, H, W)
        B, C, D, H, W = soft.size()
        scale_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)  # (B*D*H*W, C)
        scale_soft = soft.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
        p_t = F.softmax(scale_soft / self.temperature, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature ** 2)
        return loss


## CWD Loss

class ChannelNorm3D(nn.Module):
    def __init__(self):
        super(ChannelNorm3D, self).__init__()
    def forward(self, featmap):
        n, c, d, h, w = featmap.shape
        featmap = featmap.view(n, c, -1)  # (n, c, D*H*W)
        featmap = featmap.softmax(dim=-1)
        return featmap


class CWD_Loss(nn.Module):
    def __init__(self, s_channels, t_channels, norm_type='none', divergence='mse', temperature=1.0):
        super(CWD_Loss, self).__init__()

        # normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm3D()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        # loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
        else:
            raise ValueError("Unsupported divergence type.")

        self.divergence = divergence
        self.temperature = temperature
        self.conv = nn.Conv3d(s_channels, t_channels, kernel_size=1, bias=False)

    def forward(self, preds_S, preds_T):
        n, c, d, h, w = preds_S.shape

        # channel alignment
        if preds_S.size(1) != preds_T.size(1):
            preds_S = self.conv(preds_S)

        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S / self.temperature
            norm_t = preds_T.detach() / self.temperature

        if self.divergence == 'kl':
            norm_s = norm_s.log()

        loss = self.criterion(norm_s, norm_t)

        if self.norm_type in ['channel', 'channel_mean']:
            loss /= n * c
        else:
            loss /= n * d * h * w

        return loss * (self.temperature ** 2)



# class IFVD_Loss(nn.Module):
#     def __init__(self, student_channels: int, teacher_channels: int, num_classes):
#         super(IFVD_Loss, self).__init__()
#         self.num_classes = num_classes
#         self.mse = nn.MSELoss()
#         self.cos = nn.CosineSimilarity(dim=1)
#         self.channel_adapter = nn.Conv3d(student_channels, teacher_channels, kernel_size=1, bias=False)


#     def forward(self, feat_S, feat_T, target):
#         """
#         feat_S, feat_T: (N, C, D, H, W)
#         target: (N, 1, D, H, W) - assumed from nnUNetv2
#         """
#         if feat_S.size(1) != feat_T.size(1):
#             feat_S = self.channel_adapter(feat_S)
#         N, C, D, H, W = feat_S.shape

#         # Optional resizing
#         if target.shape[2:] != feat_S.shape[2:]:
#             target = F.interpolate(target.float(), size=(D, H, W), mode='nearest')
#         else:
#             target = target.float()

#         tar_feat_S = target.expand(-1, C, -1, -1, -1)
#         tar_feat_T = target.expand(-1, C, -1, -1, -1)

#         center_feat_S = feat_S.clone()
#         center_feat_T = feat_T.clone()

#         for i in range(self.num_classes):
#             mask_feat_S = (tar_feat_S == i).float()  # (N, C, D, H, W)
#             mask_feat_T = (tar_feat_T == i).float()

#             # Mean over all spatial positions per class
#             sum_mask_S = mask_feat_S.sum(dim=[2, 3, 4], keepdim=True) + 1e-6
#             sum_mask_T = mask_feat_T.sum(dim=[2, 3, 4], keepdim=True) + 1e-6

#             mean_feat_S = (mask_feat_S * feat_S).sum(dim=[2, 3, 4], keepdim=True) / sum_mask_S
#             mean_feat_T = (mask_feat_T * feat_T).sum(dim=[2, 3, 4], keepdim=True) / sum_mask_T

#             center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * mean_feat_S
#             center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * mean_feat_T

#         pcsim_feat_S = self.cos(feat_S, center_feat_S)
#         pcsim_feat_T = self.cos(feat_T, center_feat_T)

#         loss = self.mse(pcsim_feat_S, pcsim_feat_T)
#         return loss

class IFVD_Loss(nn.Module):
    """
    IFVD loss (ECCV'20) adapted to 3D features.
    Accepts targets in either (N,1,D,H,W) integer labels or (N,K,D,H,W) one-hot.
    """
    def __init__(self, student_channels: int, teacher_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=1)
        self.channel_adapter = nn.Conv3d(student_channels, teacher_channels, kernel_size=1, bias=False)

    @torch.no_grad()
    def _resize_target(self, target, size_3d):
        # target: (N,1,...) or (N,K,...)
        if target.shape[2:] != size_3d:
            target = F.interpolate(target.float(), size=size_3d, mode='nearest')
        return target

    def _iter_class_masks(self, target, K, spatial_size, device):
        """
        Yield per-class binary masks of shape (N,1,D,H,W) as float.
        Handles:
          - (N,1,D,H,W) integer labels in [0..K-1]
          - (N,K,D,H,W) one-hot masks (0/1)
        """
        N = target.size(0)
        D, H, W = spatial_size

        if target.size(1) == 1:
            # integer labels
            labels = target.long().squeeze(1)  # (N,D,H,W)
            for i in range(K):
                mask = (labels == i).unsqueeze(1).float()  # (N,1,D,H,W)
                yield mask.to(device)
        elif target.size(1) == K:
            # one-hot
            for i in range(K):
                # ensure binary float mask with explicit channel dim = 1
                mask = target[:, i:i+1, ...].float()        # (N,1,D,H,W)
                yield mask.to(device)
        else:
            raise ValueError(
                f"Target channel dim = {target.size(1)} not in {{1, K={K}}}. "
                "Please pass integer labels (N,1,...) or one-hot (N,K,...)."
            )

    def forward(self, feat_S, feat_T, target):
        """
        feat_S, feat_T: (N, C, D, H, W)
        target: (N,1,D,H,W) integer labels OR (N,K,D,H,W) one-hot
        """
        # channel align S->T if needed
        if feat_S.size(1) != feat_T.size(1):
            feat_S = self.channel_adapter(feat_S)
        N, C, D, H, W = feat_S.shape
        device = feat_S.device

        # resize target if spatial mismatched; keep channel dim
        target = self._resize_target(target, (D, H, W))

        # initialize "center features" as a copy
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()

        eps = 1e-6
        # iterate per-class masks robustly (supports 1-ch labels and K-ch one-hot)
        for mask_1c in self._iter_class_masks(target, self.num_classes, (D, H, W), device):
            # expand to feature channels: (N,C,D,H,W)
            mask_c = mask_1c.expand(-1, C, -1, -1, -1)

            # per-class means over spatial dims
            sum_mask = mask_c.sum(dim=(2, 3, 4), keepdim=True) + eps
            mean_S = (mask_c * feat_S).sum(dim=(2, 3, 4), keepdim=True) / sum_mask
            mean_T = (mask_c * feat_T).sum(dim=(2, 3, 4), keepdim=True) / sum_mask

            # fill masked positions with the class centers
            center_feat_S = torch.where(mask_c.bool(), mean_S, center_feat_S)
            center_feat_T = torch.where(mask_c.bool(), mean_T, center_feat_T)

        # cosine similarity maps over channel dim -> (N,D,H,W)
        pcsim_feat_S = self.cos(feat_S, center_feat_S)
        pcsim_feat_T = self.cos(feat_T, center_feat_T)

        return self.mse(pcsim_feat_S, pcsim_feat_T)

class SKD_Loss(nn.Module):
    def __init__(self, patch_size: int = 2, eps: float = 1e-6):
        """
        目标: MSE( X^T X , Y^T Y )，用 Gram 恒等式：
        ||X^T X - Y^T Y||_F^2 = ||X X^T||_F^2 + ||Y Y^T||_F^2 - 2 * tr((X X^T)(Y Y^T))
        - 仅对齐通道维(C)；不做 D/H/W 对齐（假定一致）
        - 无分块/无双循环；显存与时间主要随 C、P 线性增长
        """
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps
        self.pool3d = nn.MaxPool3d(kernel_size=patch_size, stride=patch_size, padding=0, ceil_mode=False)
        self.proj: nn.Module | None = None  # 懒初始化: C_s -> C_t

    def _ensure_proj(self, c_s: int, c_t: int, device: torch.device):
        if isinstance(self.proj, nn.Conv3d) and self.proj.in_channels == c_s and self.proj.out_channels == c_t:
            return
        self.proj = nn.Identity() if c_s == c_t else nn.Conv3d(c_s, c_t, kernel_size=1, bias=False).to(device)

    def forward(self, feat_S: torch.Tensor, feat_T: torch.Tensor) -> torch.Tensor:
        # 1) 统一下采样（仅改变 P）
        feat_S = self.pool3d(feat_S)
        feat_T = self.pool3d(feat_T)

        # 2) 通道对齐（仅 C）
        c_s, c_t = feat_S.size(1), feat_T.size(1)
        self._ensure_proj(c_s, c_t, feat_S.device)
        if not isinstance(self.proj, nn.Identity):
            feat_S = self.proj(feat_S)
        N, C = feat_T.size(0), feat_T.size(1)

        # 3) 通道归一化（稳健 eps）
        feat_S = F.normalize(feat_S, p=2, dim=1, eps=self.eps)
        feat_T = F.normalize(feat_T, p=2, dim=1, eps=self.eps)
        feat_S = torch.nan_to_num(feat_S)
        feat_T = torch.nan_to_num(feat_T)

        # 4) 展平 -> (N, C, P)
        XS = feat_S.contiguous().view(N, C, -1).float()
        XT = feat_T.contiguous().view(N, C, -1).float()
        P = XS.size(-1)

        # 5) 用 Gram 恒等式在 fp32 下一次完成计算（无双循环）
        with torch.cuda.amp.autocast(enabled=False):
            Gs = torch.bmm(XS, XS.transpose(1, 2))   # (N, C, C) = X X^T
            Gt = torch.bmm(XT, XT.transpose(1, 2))   # (N, C, C) = Y Y^T

            # ||Gs||_F^2 与 ||Gt||_F^2
            n1 = (Gs * Gs).sum(dim=(1, 2))           # (N,)
            n2 = (Gt * Gt).sum(dim=(1, 2))           # (N,)

            # tr(Gs @ Gt)
            cross = torch.bmm(Gs, Gt).diagonal(dim1=1, dim2=2).sum(1)  # (N,)

            ssd = (n1 + n2 - 2.0 * cross).sum()      # 标量
            mse = ssd / float(N * P * P)

        return mse.to(feat_T.dtype)


# class SKD_Loss(nn.Module):
#     def __init__(self, patch_size: int = 2):
#         """
#         Structural Knowledge Distillation Loss (3D version)
#         Args:
#             patch_size (int): pooling kernel size for downsampling feature maps
#         """
#         super(SKD_Loss, self).__init__()
#         self.patch_size = patch_size

#     def pair_wise_sim_map(self, feat):
#         """
#         Compute pairwise similarity matrix of feature maps
#         Input: feat (N, C, D, H, W)
#         Output: sim_map (N, D*H*W, D*H*W)
#         """
#         N, C, D, H, W = feat.shape
#         feat = feat.view(N, C, -1)  # (N, C, D*H*W)
#         feat_T = feat.transpose(1, 2)  # (N, D*H*W, C)
#         sim_map = torch.bmm(feat_T, feat)  # (N, D*H*W, D*H*W)
#         return sim_map

#     def forward(self, feat_S, feat_T):
#         """
#         feat_S, feat_T: (N, C, D, H, W)
#         """
#         # Downsample with 3D max pooling
#         pool3d = nn.MaxPool3d(kernel_size=self.patch_size, stride=self.patch_size, padding=0, ceil_mode=True)
#         feat_S = pool3d(feat_S)
#         feat_T = pool3d(feat_T)

#         # Normalize along channel dimension
#         feat_S = F.normalize(feat_S, p=2, dim=1)
#         feat_T = F.normalize(feat_T, p=2, dim=1)

#         # Compute similarity maps
#         sim_map_S = self.pair_wise_sim_map(feat_S)  # (N, P, P)
#         sim_map_T = self.pair_wise_sim_map(feat_T)  # (N, P, P)

#         # Compute mean squared error between similarity maps
#         loss = F.mse_loss(sim_map_S, sim_map_T)
#         return loss


class PSD3D(nn.Module):
    def __init__(self):
        super(PSD3D, self).__init__()
        self.p = 2

    def attention_preprocess(self, f_list):
        outs = []
        for f in f_list:
            # f: (B, C, D, H, W)
            att = f.pow(self.p).mean(dim=1)  # (B, D, H, W)
            att = att.view(f.size(0), -1)    # (B, D*H*W)
            att = F.normalize(att, dim=1)
            outs.append(att)
        return outs

    def residual_attention(self, f_list):
        ra_list = []
        for n in range(len(f_list) - 1):
            for m in range(n + 1, len(f_list)):
                f_n = f_list[n]
                f_m = f_list[m]

                # Resize f_m to match f_n if necessary
                if f_m.shape != f_n.shape:
                    f_m = F.interpolate(f_m.unsqueeze(1), size=f_n.shape[1:], mode='linear', align_corners=False).squeeze(1)

                ra = F.normalize(f_m - f_n, dim=1)
                ra_list.append(ra)
        return ra_list

    def forward(self, feat_S_list, feat_T_list):
        feat_S_list = self.attention_preprocess(feat_S_list)
        feat_T_list = self.attention_preprocess(feat_T_list)

        ra_S_list = self.residual_attention(feat_S_list)
        ra_T_list = self.residual_attention(feat_T_list)

        K = len(ra_S_list)
        psd_loss = torch.tensor(0.).to(feat_S_list[0].device)
        for k in range(K):
            psd_loss += (F.normalize(ra_S_list[k]) - F.normalize(ra_T_list[k])).pow(2).mean()

        psd_loss = psd_loss / K
        return psd_loss


class CSD3D(nn.Module):
    def __init__(self, s_channels=None, t_channels=None, tau=2.0):
        super(CSD3D, self).__init__()
        self.pooling = nn.AvgPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.tau = tau
        if s_channels != t_channels:
            self.proj = nn.Conv3d(s_channels, t_channels, kernel_size=1, bias=False)
        else:
            self.proj = None

    def pair_wise_sim_map(self, fea):
        B, C, D, H, W = fea.size()
        fea = fea.view(B, C, -1)  # flatten spatial dims
        fea = F.softmax(fea / self.tau, dim=-1)
        sim_map = torch.bmm(fea, fea.transpose(1, 2))  # (B, C, C)
        return sim_map

    def forward(self, feat_S, feat_T):
        feat_S = self.pooling(feat_S)
        feat_T = self.pooling(feat_T)

        if self.proj is not None:
            feat_S = self.proj(feat_S)

        S_sim_map = self.pair_wise_sim_map(feat_S)
        T_sim_map = self.pair_wise_sim_map(feat_T)

        sim_dis = (S_sim_map - T_sim_map).pow(2).mean()
        return sim_dis


class DoubleSimKD_Loss(nn.Module):
    def __init__(self, s_channels_last, t_channels_last):
        super(DoubleSimKD_Loss, self).__init__()
        self.psd = PSD3D()
        self.csd = CSD3D(s_channels=s_channels_last, t_channels=t_channels_last)

    def forward(self, feat_S_list, feat_T_list):
        psd_loss = self.psd(feat_S_list, feat_T_list)
        csd_loss = self.csd(feat_S_list[-1], feat_T_list[-1])
        return psd_loss, csd_loss


class AT_Loss(nn.Module):
    def __init__(self, s_channels: int = None, t_channels: int = None):
        super(AT_Loss, self).__init__()
        self.p = 2
        if s_channels is not None and t_channels is not None and s_channels != t_channels:
            self.proj = nn.Conv3d(s_channels, t_channels, kernel_size=1, bias=False)
        else:
            self.proj = None

    def at(self, f):
        # f: [N, C, D, H, W]
        # return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
        return f.pow(self.p).mean(1).view(f.size(0), -1)

    def forward(self, feat_S, feat_T):
        # 输入也为 [N, C, D, H, W]
        loss = (self.at(feat_S) - self.at(feat_T)).pow(2).mean()
        return loss
    

class CIRKD_Loss(nn.Module):
    """
    CIRKD Loss for 3D segmentation (MRI/CT) with channel projection
    and adaptive pooling to fixed resolution.

    - Projects student features to teacher's channel dimension.
    - L2-normalizes features along channel dimension.
    - Uses adaptive_avg_pool3d to unify spatial resolution for all stages.
    - Computes voxel-to-voxel similarity matrices between all sample pairs (i, j).
    - Uses KL divergence to match similarity distributions.
    - No logits distillation, only relation distillation.
    """

    def __init__(self, s_channels, t_channels, temperature=0.1, target_size=(8, 8, 8)):
        """
        Args:
            s_channels (int): Number of channels in student feature maps.
            t_channels (int): Number of channels in teacher feature maps.
            temperature (float): Softmax temperature for similarity distribution smoothing.
            target_size (tuple): Output spatial resolution (D, H, W) after adaptive pooling.
        """
        super(CIRKD_Loss, self).__init__()
        self.temperature = temperature
        self.target_size = target_size

        # Projector to align student's channels to teacher's channel dimension
        self.projector = nn.Sequential(
            nn.Conv3d(s_channels, t_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(t_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(t_channels, t_channels, kernel_size=1, bias=False)
        )

    def pair_wise_sim_map(self, fea_0, fea_1):
        """
        Compute voxel-to-voxel similarity map between two 3D feature volumes.

        Args:
            fea_0, fea_1: Feature tensors of shape (C, D, H, W), already L2-normalized.
        Returns:
            sim_map: Similarity matrix of shape (N, N), where N = D*H*W.
        """
        C, D, H, W = fea_0.size()
        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)  # (N, C)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)  # (N, C)
        sim_map = torch.mm(fea_0, fea_1.transpose(0, 1))  # (N, N)
        return sim_map

    def forward(self, feat_S, feat_T):
        """
        Args:
            feat_S: Student features (B, C_s, D, H, W)
            feat_T: Teacher features (B, C_t, D, H, W)
        Returns:
            sim_loss: CIRKD relation distillation loss (scalar)
        """
        B = feat_S.size(0)

        # Step 1: Project student features to teacher's channel dimension
        feat_S = self.projector(feat_S)

        # Step 2: Adaptive average pooling to fixed resolution
        feat_S = F.adaptive_avg_pool3d(feat_S, self.target_size)
        feat_T = F.adaptive_avg_pool3d(feat_T, self.target_size)

        # Step 3: L2-normalize features along channel dimension
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        # Step 4: Compute relation distillation loss
        sim_loss = torch.tensor(0., device=feat_S.device)
        for i in range(B):
            for j in range(B):
                s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[j])
                t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[j])

                p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
                p_t = F.softmax(t_sim_map / self.temperature, dim=1)

                sim_loss += F.kl_div(p_s, p_t, reduction='batchmean')

        # Step 5: Average over all (i, j) pairs
        sim_loss = sim_loss / (B * B)

        return sim_loss
