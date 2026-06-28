from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_ndim(kernel_sizes: Sequence[Sequence[int]], strides: Sequence[Sequence[int]]) -> int:
    if kernel_sizes:
        return len(kernel_sizes[0])
    if strides:
        return len(strides[0])
    return 3


def _ensure_tuple(value: Union[int, Sequence[int]], ndim: int) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),) * ndim


def _infer_spatial_ndim(
    conv_op: Type[nn.Module],
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
) -> int:
    if isinstance(kernel_size, (list, tuple)):
        return len(kernel_size)
    if isinstance(stride, (list, tuple)):
        return len(stride)
    try:
        if issubclass(conv_op, nn.Conv3d):
            return 3
        if issubclass(conv_op, nn.Conv2d):
            return 2
    except TypeError:
        pass
    return 3


def _expand_to_list(value: Union[int, Sequence[int]], length: int, name: str) -> List[int]:
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"{name} length {len(value)} does not match n_stages {length}")
        return [int(v) for v in value]
    return [int(value)] * length


def _make_norm(norm_op: Optional[Type[nn.Module]], num_features: int, norm_op_kwargs: Optional[dict]) -> nn.Module:
    if norm_op is None:
        return nn.Identity()
    kwargs = norm_op_kwargs or {}
    return norm_op(num_features, **kwargs)


def _make_nonlin(nonlin: Optional[Type[nn.Module]], nonlin_kwargs: Optional[dict]) -> nn.Module:
    if nonlin is None:
        return nn.ReLU(inplace=True)
    kwargs = nonlin_kwargs or {}
    return nonlin(**kwargs)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        bias: bool,
        norm_op: Optional[Type[nn.Module]],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
        dropout_op: Optional[Type[nn.Module]],
        dropout_op_kwargs: Optional[dict],
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        ndim = _infer_spatial_ndim(conv_op, kernel_size, stride)
        kernel_size = _ensure_tuple(kernel_size, ndim)
        stride = _ensure_tuple(stride, ndim)
        padding = tuple(k // 2 for k in kernel_size)

        self.conv = conv_op(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        self.norm = _make_norm(norm_op, out_channels, norm_op_kwargs)
        self.act = _make_nonlin(nonlin, nonlin_kwargs) if act else nn.Identity()
        if dropout_op is None:
            self.dropout = nn.Identity()
        else:
            self.dropout = dropout_op(**(dropout_op_kwargs or {}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        conv_op: Type[nn.Module],
        channels: int,
        se_ratio: float,
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
    ):
        super().__init__()
        squeeze_channels = max(1, int(channels * se_ratio))
        self.fc1 = conv_op(channels, squeeze_channels, kernel_size=1, bias=True)
        self.act = _make_nonlin(nonlin, nonlin_kwargs)
        self.fc2 = conv_op(squeeze_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, x.dim()))
        scale = x.mean(dim=dims, keepdim=True)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class MBConvLite(nn.Module):
    def __init__(
        self,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        expand_ratio: float,
        se_ratio: float,
        bias: bool,
        norm_op: Optional[Type[nn.Module]],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
        dropout_op: Optional[Type[nn.Module]],
        dropout_op_kwargs: Optional[dict],
    ):
        super().__init__()
        mid_channels = max(in_channels, int(in_channels * expand_ratio))
        ndim = _infer_spatial_ndim(conv_op, kernel_size, stride)
        stride_tuple = _ensure_tuple(stride, ndim)
        self.use_residual = stride_tuple == tuple(1 for _ in stride_tuple) and in_channels == out_channels

        self.expand = None
        if mid_channels != in_channels:
            self.expand = ConvNormAct(
                conv_op,
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                groups=1,
                act=True,
            )

        self.depthwise = ConvNormAct(
            conv_op,
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            groups=mid_channels,
            act=True,
        )

        self.se = None
        if se_ratio and se_ratio > 0:
            self.se = SqueezeExcite(conv_op, mid_channels, se_ratio, nonlin, nonlin_kwargs)

        self.project = ConvNormAct(
            conv_op,
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            groups=1,
            act=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.expand is not None:
            x = self.expand(x)
        x = self.depthwise(x)
        if self.se is not None:
            x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x


class MobileEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_blocks_per_stage: Sequence[int],
        conv_op: Type[nn.Module],
        conv_bias: bool,
        norm_op: Optional[Type[nn.Module]],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
        dropout_op: Optional[Type[nn.Module]],
        dropout_op_kwargs: Optional[dict],
        expand_ratio: float,
        se_ratio: float,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        self.output_channels: List[int] = []

        in_channels = input_channels
        for stage_idx in range(n_stages):
            out_channels = int(features_per_stage[stage_idx])
            blocks = []
            for block_idx in range(n_blocks_per_stage[stage_idx]):
                stride = strides[stage_idx] if block_idx == 0 else [1] * len(strides[stage_idx])
                blocks.append(
                    MBConvLite(
                        conv_op=conv_op,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[stage_idx],
                        stride=stride,
                        expand_ratio=expand_ratio,
                        se_ratio=se_ratio,
                        bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                    )
                )
                in_channels = out_channels
            self.stages.append(nn.Sequential(*blocks))
            self.output_channels.append(out_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class MobileUNetV3Decoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_conv_per_stage_decoder: Sequence[int],
        conv_op: Type[nn.Module],
        conv_bias: bool,
        norm_op: Optional[Type[nn.Module]],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
        dropout_op: Optional[Type[nn.Module]],
        dropout_op_kwargs: Optional[dict],
        expand_ratio: float,
        se_ratio: float,
        deep_supervision: bool,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.stages = nn.ModuleList()
        self.seg_heads = nn.ModuleList()
        self._ndim = _infer_ndim(kernel_sizes, strides)
        self._up_mode = "trilinear" if self._ndim == 3 else "bilinear"

        for stage_idx in range(n_stages - 1):
            in_channels = int(features_per_stage[stage_idx + 1]) + int(features_per_stage[stage_idx])
            out_channels = int(features_per_stage[stage_idx])
            blocks = []
            for conv_idx in range(n_conv_per_stage_decoder[stage_idx]):
                blocks.append(
                    MBConvLite(
                        conv_op=conv_op,
                        in_channels=in_channels if conv_idx == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[stage_idx],
                        stride=1,
                        expand_ratio=expand_ratio,
                        se_ratio=se_ratio,
                        bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            self.seg_heads.append(conv_op(out_channels, num_classes, kernel_size=1, bias=True))

    def forward(self, encoder_outputs: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = encoder_outputs[-1]
        seg_outputs: List[torch.Tensor] = []

        for stage_idx in reversed(range(len(encoder_outputs) - 1)):
            skip = encoder_outputs[stage_idx]
            x = F.interpolate(x, size=skip.shape[2:], mode=self._up_mode, align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.stages[stage_idx](x)
            seg_outputs.append(self.seg_heads[stage_idx](x))

        seg_outputs = list(reversed(seg_outputs))
        if self.deep_supervision:
            return seg_outputs
        return seg_outputs[0]


class MobileUNetV3(nn.Module):
    """
    Lightweight UNet-style backbone with MobileNetV3-like blocks.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        conv_op: Type[nn.Module],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_conv_per_stage_decoder: Union[int, Sequence[int]],
        n_blocks_per_stage: Optional[Union[int, Sequence[int]]] = None,
        n_conv_per_stage: Optional[Union[int, Sequence[int]]] = None,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = True,
        expand_ratio: float = 2.0,
        se_ratio: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        if len(features_per_stage) != n_stages:
            raise ValueError("features_per_stage length must match n_stages")

        kernel_sizes = [list(k) for k in kernel_sizes]
        strides = [list(s) for s in strides]
        if n_conv_per_stage_decoder is None:
            raise ValueError("MobileUNetV3 requires n_conv_per_stage_decoder")
        if n_blocks_per_stage is None:
            n_blocks_per_stage = n_conv_per_stage
        if n_blocks_per_stage is None:
            raise ValueError("MobileUNetV3 requires n_blocks_per_stage or n_conv_per_stage")
        n_blocks_per_stage = _expand_to_list(n_blocks_per_stage, n_stages, "n_blocks_per_stage")

        if isinstance(n_conv_per_stage_decoder, (list, tuple)):
            if len(n_conv_per_stage_decoder) == n_stages:
                n_conv_per_stage_decoder = list(n_conv_per_stage_decoder[:-1])
            elif len(n_conv_per_stage_decoder) != n_stages - 1:
                raise ValueError("n_conv_per_stage_decoder length must match n_stages - 1")
            n_conv_per_stage_decoder = [int(v) for v in n_conv_per_stage_decoder]
        else:
            n_conv_per_stage_decoder = [int(n_conv_per_stage_decoder)] * (n_stages - 1)

        self.encoder = MobileEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_op=conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            expand_ratio=expand_ratio,
            se_ratio=se_ratio,
        )

        self.decoder = MobileUNetV3Decoder(
            num_classes=num_classes,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_op=conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            expand_ratio=expand_ratio,
            se_ratio=se_ratio,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        encoder_outputs = self.encoder(x)
        return self.decoder(encoder_outputs)


__all__ = ["MobileUNetV3"]
