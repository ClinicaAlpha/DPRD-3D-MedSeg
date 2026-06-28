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


def _channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    if x.shape[1] % groups != 0:
        raise ValueError(f"channels {x.shape[1]} not divisible by groups {groups}")
    batch_size = x.shape[0]
    channels_per_group = x.shape[1] // groups
    spatial = x.shape[2:]
    x = x.view(batch_size, groups, channels_per_group, *spatial)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels_per_group * groups, *spatial)


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


class ShuffleV2Block(nn.Module):
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
    ):
        super().__init__()
        ndim = _infer_spatial_ndim(conv_op, kernel_size, stride)
        self.stride = _ensure_tuple(stride, ndim)
        self.is_downsample = any(s > 1 for s in self.stride)

        if out_channels % 2 != 0:
            raise ValueError("ShuffleV2Block requires out_channels to be divisible by 2")

        branch_channels = out_channels // 2
        if not self.is_downsample:
            if in_channels != out_channels:
                raise ValueError("ShuffleV2Block stride=1 requires in_channels == out_channels")
            if in_channels % 2 != 0:
                raise ValueError("ShuffleV2Block stride=1 requires in_channels divisible by 2")

            self.branch1 = nn.Identity()
            self.branch2 = nn.Sequential(
                ConvNormAct(
                    conv_op,
                    branch_channels,
                    branch_channels,
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
                ),
                ConvNormAct(
                    conv_op,
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    bias=bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    groups=branch_channels,
                    act=True,
                ),
                ConvNormAct(
                    conv_op,
                    branch_channels,
                    branch_channels,
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
                ),
            )
        else:
            self.branch1 = nn.Sequential(
                ConvNormAct(
                    conv_op,
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    bias=bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    groups=in_channels,
                    act=True,
                ),
                ConvNormAct(
                    conv_op,
                    in_channels,
                    branch_channels,
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
                ),
            )
            self.branch2 = nn.Sequential(
                ConvNormAct(
                    conv_op,
                    in_channels,
                    branch_channels,
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
                ),
                ConvNormAct(
                    conv_op,
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    bias=bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    groups=branch_channels,
                    act=True,
                ),
                ConvNormAct(
                    conv_op,
                    branch_channels,
                    branch_channels,
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
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_downsample:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        return _channel_shuffle(out, groups=2)


class ShuffleNetV2Encoder(nn.Module):
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
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        self.output_channels: List[int] = []

        self.stem: Optional[nn.Module] = None
        in_channels = input_channels
        if n_stages > 0:
            first_stride = strides[0]
            if all(int(s) == 1 for s in first_stride) and int(features_per_stage[0]) != input_channels:
                self.stem = ConvNormAct(
                    conv_op=conv_op,
                    in_channels=input_channels,
                    out_channels=int(features_per_stage[0]),
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    groups=1,
                    act=True,
                )
                in_channels = int(features_per_stage[0])
        for stage_idx in range(n_stages):
            out_channels = int(features_per_stage[stage_idx])
            blocks = []
            for block_idx in range(n_blocks_per_stage[stage_idx]):
                stride = strides[stage_idx] if block_idx == 0 else [1] * len(strides[stage_idx])
                blocks.append(
                    ShuffleV2Block(
                        conv_op=conv_op,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[stage_idx],
                        stride=stride,
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
        if self.stem is not None:
            x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class ShuffleNetV2Decoder(nn.Module):
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
                    ConvNormAct(
                        conv_op=conv_op,
                        in_channels=in_channels if conv_idx == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[stage_idx],
                        stride=1,
                        bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        groups=1,
                        act=True,
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


class ShuffleNetV2UNet(nn.Module):
    """
    ShuffleNetV2 encoder + nnUNet-style decoder.
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
        **kwargs,
    ):
        super().__init__()
        if len(features_per_stage) != n_stages:
            raise ValueError("features_per_stage length must match n_stages")

        kernel_sizes = [list(k) for k in kernel_sizes]
        strides = [list(s) for s in strides]
        if n_conv_per_stage_decoder is None:
            raise ValueError("ShuffleNetV2UNet requires n_conv_per_stage_decoder")
        if n_blocks_per_stage is None:
            n_blocks_per_stage = n_conv_per_stage
        if n_blocks_per_stage is None:
            raise ValueError("ShuffleNetV2UNet requires n_blocks_per_stage or n_conv_per_stage")
        n_blocks_per_stage = _expand_to_list(n_blocks_per_stage, n_stages, "n_blocks_per_stage")

        if isinstance(n_conv_per_stage_decoder, (list, tuple)):
            if len(n_conv_per_stage_decoder) == n_stages:
                n_conv_per_stage_decoder = list(n_conv_per_stage_decoder[:-1])
            elif len(n_conv_per_stage_decoder) != n_stages - 1:
                raise ValueError("n_conv_per_stage_decoder length must match n_stages - 1")
            n_conv_per_stage_decoder = [int(v) for v in n_conv_per_stage_decoder]
        else:
            n_conv_per_stage_decoder = [int(n_conv_per_stage_decoder)] * (n_stages - 1)

        self.encoder = ShuffleNetV2Encoder(
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
        )

        self.decoder = ShuffleNetV2Decoder(
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
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        encoder_outputs = self.encoder(x)
        return self.decoder(encoder_outputs)


__all__ = ["ShuffleNetV2UNet"]
