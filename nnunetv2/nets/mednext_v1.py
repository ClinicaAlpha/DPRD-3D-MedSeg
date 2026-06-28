from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def _infer_ndim_from_conv(
    conv_op: Optional[Type[nn.Module]],
    kernel_sizes: Optional[Sequence[Sequence[int]]],
    strides: Optional[Sequence[Sequence[int]]],
) -> int:
    if conv_op is not None:
        try:
            if issubclass(conv_op, nn.Conv3d):
                return 3
            if issubclass(conv_op, nn.Conv2d):
                return 2
        except TypeError:
            pass
    if kernel_sizes:
        return len(kernel_sizes[0])
    if strides:
        return len(strides[0])
    return 3


def _to_stage_nd_list(
    value: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]],
    n_stages: int,
    ndim: int,
    name: str,
    default: int,
) -> List[Tuple[int, ...]]:
    if value is None:
        return [tuple([default] * ndim) for _ in range(n_stages)]

    if isinstance(value, int):
        return [tuple([int(value)] * ndim) for _ in range(n_stages)]

    values = list(value)
    if not values:
        return [tuple([default] * ndim) for _ in range(n_stages)]

    # flat ndim vector -> replicate to all stages
    if len(values) == ndim and all(isinstance(v, (int, float)) for v in values):
        item = tuple(int(v) for v in values)
        return [item for _ in range(n_stages)]

    out: List[Tuple[int, ...]] = []
    for idx, item in enumerate(values):
        if isinstance(item, (list, tuple)):
            if len(item) != ndim:
                raise ValueError(f"{name}[{idx}] must have {ndim} entries, got {len(item)}")
            out.append(tuple(int(v) for v in item))
        else:
            out.append(tuple([int(item)] * ndim))

    if len(out) < n_stages:
        out.extend([out[-1]] * (n_stages - len(out)))
    return out[:n_stages]


def _expand_to_length(value: Optional[Union[int, Sequence[int]]], length: int, name: str) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)] * length
    out = [int(v) for v in value]
    if not out:
        return None
    if len(out) < length:
        out.extend([out[-1]] * (length - len(out)))
    return out[:length]


def _adapt_legacy_symmetric(values: Sequence[int], n_stages: int) -> List[int]:
    vals = [int(v) for v in values]
    if len(vals) < 9:
        vals.extend([vals[-1]] * (9 - len(vals)))
    legacy_enc = vals[:4]
    legacy_bottleneck = vals[4]

    enc_depth = max(n_stages - 1, 0)
    if enc_depth <= len(legacy_enc):
        enc = legacy_enc[:enc_depth]
    else:
        enc = legacy_enc + [legacy_enc[-1]] * (enc_depth - len(legacy_enc))

    full_enc = enc + [legacy_bottleneck]
    full_dec = list(reversed(enc))
    return full_enc + full_dec


def _resolve_expansion(exp_r: Union[int, Sequence[int]], n_stages: int) -> List[int]:
    total = 2 * n_stages - 1
    if isinstance(exp_r, int):
        return [int(exp_r)] * total
    raw = [int(v) for v in exp_r]
    if not raw:
        return [4] * total
    if len(raw) == total:
        return raw
    if len(raw) == n_stages:
        return raw + list(reversed(raw[:-1]))
    if len(raw) == 9:
        return _adapt_legacy_symmetric(raw, n_stages)
    if len(raw) < total:
        raw.extend([raw[-1]] * (total - len(raw)))
    return raw[:total]


def _resolve_block_counts(
    n_stages: int,
    n_blocks_per_stage: Optional[Union[int, Sequence[int]]],
    n_conv_per_stage: Optional[Union[int, Sequence[int]]],
    n_conv_per_stage_decoder: Optional[Union[int, Sequence[int]]],
    block_counts: Optional[Sequence[int]],
    preset_counts: Optional[Sequence[int]],
) -> Tuple[List[int], List[int]]:
    if block_counts is not None:
        raw = [int(v) for v in block_counts]
        if len(raw) == (2 * n_stages - 1):
            return raw[:n_stages], raw[n_stages:]
        if len(raw) == n_stages:
            enc = raw
            return enc, list(reversed(enc[:-1]))
        if len(raw) == 9:
            merged = _adapt_legacy_symmetric(raw, n_stages)
            return merged[:n_stages], merged[n_stages:]
        raise ValueError(
            f"block_counts must have length {n_stages}, {2 * n_stages - 1}, or 9. Got {len(raw)}"
        )

    source = n_blocks_per_stage if n_blocks_per_stage is not None else n_conv_per_stage
    enc_counts = _expand_to_length(source, n_stages, "n_blocks_per_stage")

    if enc_counts is None and preset_counts is not None:
        merged = _adapt_legacy_symmetric([int(v) for v in preset_counts], n_stages)
        enc_counts = merged[:n_stages]

    if enc_counts is None:
        enc_counts = [2] * n_stages

    if n_conv_per_stage_decoder is None:
        dec_counts = list(reversed(enc_counts[:-1]))
    else:
        if isinstance(n_conv_per_stage_decoder, int):
            dec_counts = [int(n_conv_per_stage_decoder)] * (n_stages - 1)
        else:
            dec_counts = [int(v) for v in n_conv_per_stage_decoder]
            if len(dec_counts) == n_stages:
                dec_counts = dec_counts[:-1]
            if len(dec_counts) < (n_stages - 1):
                dec_counts.extend([dec_counts[-1]] * ((n_stages - 1) - len(dec_counts)))
            dec_counts = dec_counts[: n_stages - 1]

    return enc_counts, dec_counts


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor, dummy_tensor: bool = False) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        shape = (self.weight.shape[0],) + (1,) * (x.ndim - 2)
        weight = self.weight.view(shape)
        bias = self.bias.view(shape)
        return weight * x + bias


class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: Union[int, Sequence[int]] = 3,
        do_res: bool = True,
        norm_type: str = "group",
        n_groups: Optional[int] = None,
        dim: str = "3d",
        grn: bool = False,
    ):
        super().__init__()
        self.do_res = do_res

        if dim not in ["2d", "3d"]:
            raise ValueError(f"Unsupported dim '{dim}'")
        self.dim = dim
        conv = nn.Conv2d if dim == "2d" else nn.Conv3d

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * (2 if dim == "2d" else 3)
        kernel_size = tuple(int(k) for k in kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=in_channels if n_groups is None else n_groups,
        )

        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        elif norm_type == "layer":
            self.norm = LayerNorm(normalized_shape=in_channels, data_format="channels_first")
        else:
            raise ValueError(f"Unsupported norm_type '{norm_type}'")

        self.conv2 = conv(in_channels=in_channels, out_channels=exp_r * in_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.conv3 = conv(in_channels=exp_r * in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.grn = grn
        if grn:
            if dim == "3d":
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
            else:
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor, dummy_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            if self.dim == "3d":
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            else:
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        do_res: bool = False,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )
        conv = nn.Conv2d if dim == "2d" else nn.Conv3d
        self.resample_do_res = do_res

        if isinstance(stride, int):
            stride = (stride,) * (2 if dim == "2d" else 3)
        stride = tuple(int(s) for s in stride)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * len(stride)
        kernel_size = tuple(int(k) for k in kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        if do_res:
            self.res_conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, x: torch.Tensor, dummy_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = super().forward(x)
        if self.resample_do_res:
            x1 = x1 + self.res_conv(x)
        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        do_res: bool = False,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )
        conv_t = nn.ConvTranspose2d if dim == "2d" else nn.ConvTranspose3d
        self.resample_do_res = do_res

        if isinstance(stride, int):
            stride = (stride,) * (2 if dim == "2d" else 3)
        stride = tuple(int(s) for s in stride)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * len(stride)
        kernel_size = tuple(int(k) for k in kernel_size)
        padding = tuple(k // 2 for k in kernel_size)
        output_padding = tuple(max(s - 1, 0) for s in stride)

        if do_res:
            self.res_conv = conv_t(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                output_padding=output_padding,
            )

        self.conv1 = conv_t(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=in_channels,
        )

    def forward(self, x: torch.Tensor, dummy_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = super().forward(x)
        if self.resample_do_res:
            x1 = x1 + self.res_conv(x)
        return x1


class MedNeXtCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features_per_stage: Sequence[int],
        n_classes: int,
        enc_kernel_sizes: Sequence[Sequence[int]],
        dec_kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        enc_block_counts: Sequence[int],
        dec_block_counts: Sequence[int],
        exp_r: Sequence[int],
        deep_supervision: bool = False,
        do_res: bool = True,
        do_res_up_down: bool = True,
        checkpoint_style: Optional[str] = None,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
    ):
        super().__init__()
        if checkpoint_style not in [None, "outside_block"]:
            raise ValueError("checkpoint_style must be None or 'outside_block'")

        self.do_ds = bool(deep_supervision)
        self.outside_block_checkpointing = checkpoint_style == "outside_block"
        self.n_stages = len(features_per_stage)

        conv = nn.Conv2d if dim == "2d" else nn.Conv3d
        self.stem = conv(in_channels, int(features_per_stage[0]), kernel_size=1)

        self.enc_stages = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        self.seg_heads = nn.ModuleList()

        exp_ptr = 0
        for i in range(self.n_stages):
            blocks = [
                MedNeXtBlock(
                    in_channels=int(features_per_stage[i]),
                    out_channels=int(features_per_stage[i]),
                    exp_r=int(exp_r[exp_ptr]),
                    kernel_size=enc_kernel_sizes[i],
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(int(enc_block_counts[i]))
            ]
            self.enc_stages.append(nn.Sequential(*blocks))
            exp_ptr += 1

        for i in range(self.n_stages - 1):
            self.down_blocks.append(
                MedNeXtDownBlock(
                    in_channels=int(features_per_stage[i]),
                    out_channels=int(features_per_stage[i + 1]),
                    exp_r=int(exp_r[i + 1]),
                    kernel_size=enc_kernel_sizes[i + 1],
                    stride=strides[i + 1],
                    do_res=do_res_up_down,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
            )

        for i in range(self.n_stages - 2, -1, -1):
            self.up_blocks.insert(
                0,
                MedNeXtUpBlock(
                    in_channels=int(features_per_stage[i + 1]),
                    out_channels=int(features_per_stage[i]),
                    exp_r=int(exp_r[exp_ptr]),
                    kernel_size=dec_kernel_sizes[i],
                    stride=strides[i + 1],
                    do_res=do_res_up_down,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                ),
            )

            dec_blocks = [
                MedNeXtBlock(
                    in_channels=int(features_per_stage[i]),
                    out_channels=int(features_per_stage[i]),
                    exp_r=int(exp_r[exp_ptr]),
                    kernel_size=dec_kernel_sizes[i],
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(int(dec_block_counts[i]))
            ]
            self.dec_stages.insert(0, nn.Sequential(*dec_blocks))
            self.seg_heads.insert(0, conv(int(features_per_stage[i]), n_classes, kernel_size=1, bias=True))
            exp_ptr += 1

        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def set_deep_supervision(self, enabled: bool) -> None:
        self.do_ds = bool(enabled)

    def iterative_checkpoint(self, sequential_block: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        for layer in sequential_block:
            x = checkpoint.checkpoint(layer, x, self.dummy_tensor)
        return x

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []

        for i in range(self.n_stages - 1):
            if self.outside_block_checkpointing:
                x_enc = self.iterative_checkpoint(self.enc_stages[i], x)
                x = checkpoint.checkpoint(self.down_blocks[i], x_enc, self.dummy_tensor)
            else:
                x_enc = self.enc_stages[i](x)
                x = self.down_blocks[i](x_enc)
            skips.append(x_enc)

        if self.outside_block_checkpointing:
            x = self.iterative_checkpoint(self.enc_stages[-1], x)
        else:
            x = self.enc_stages[-1](x)

        seg_outputs: List[torch.Tensor] = []
        for i in range(self.n_stages - 2, -1, -1):
            if self.outside_block_checkpointing:
                x_up = checkpoint.checkpoint(self.up_blocks[i], x, self.dummy_tensor)
                x = self.iterative_checkpoint(self.dec_stages[i], skips[i] + x_up)
                seg_outputs.append(checkpoint.checkpoint(self.seg_heads[i], x, self.dummy_tensor))
            else:
                x_up = self.up_blocks[i](x)
                x = self.dec_stages[i](skips[i] + x_up)
                seg_outputs.append(self.seg_heads[i](x))

        seg_outputs = list(reversed(seg_outputs))
        if self.do_ds:
            return seg_outputs
        return seg_outputs[0]


class _MedNeXtEncoderProxy(nn.Module):
    def __init__(
        self,
        stem: nn.Module,
        stages: Sequence[nn.Module],
        down_blocks: Sequence[nn.Module],
        output_channels: Sequence[int],
    ):
        super().__init__()
        self.stem = stem
        self.stages = nn.ModuleList(list(stages))
        self.down_blocks = nn.ModuleList(list(down_blocks))
        self.output_channels = [int(v) for v in output_channels]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        x = self.stem(x)
        for idx, down in enumerate(self.down_blocks):
            x = self.stages[idx](x)
            outputs.append(x)
            x = down(x)
        x = self.stages[-1](x)
        outputs.append(x)
        return outputs


class _DecoderProxy:
    def __init__(self, owner: "MedNeXtV1"):
        self._owner = owner

    @property
    def deep_supervision(self) -> bool:
        return self._owner.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, enabled: bool) -> None:
        self._owner.set_deep_supervision(enabled)


class MedNeXtV1(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        conv_op: Type[nn.Module],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_conv_per_stage_decoder: Optional[Union[int, Sequence[int]]] = None,
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
        mednext_model_id: Optional[str] = None,
        exp_r: Union[int, Sequence[int]] = 4,
        block_counts: Optional[Sequence[int]] = None,
        kernel_size: Optional[int] = None,
        enc_kernel_size: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]] = None,
        dec_kernel_size: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]] = None,
        do_res: bool = True,
        do_res_up_down: bool = True,
        checkpoint_style: Optional[str] = None,
        norm_type: str = "group",
        grn: bool = False,
        **kwargs,
    ):
        super().__init__()
        _ = (conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, kwargs)

        if len(features_per_stage) != int(n_stages):
            raise ValueError(f"features_per_stage length ({len(features_per_stage)}) must match n_stages ({n_stages})")

        ndim = _infer_ndim_from_conv(conv_op, kernel_sizes, strides)
        dim = "3d" if ndim == 3 else "2d"

        if kernel_size is not None:
            enc_kernels = _to_stage_nd_list(kernel_size, int(n_stages), ndim, "kernel_size", 3)
            dec_kernels = _to_stage_nd_list(kernel_size, int(n_stages), ndim, "kernel_size", 3)
        else:
            enc_kernels = _to_stage_nd_list(enc_kernel_size if enc_kernel_size is not None else kernel_sizes,
                                            int(n_stages), ndim, "enc_kernel_size", 3)
            dec_kernels = _to_stage_nd_list(dec_kernel_size if dec_kernel_size is not None else kernel_sizes,
                                            int(n_stages), ndim, "dec_kernel_size", 3)

        stride_list = _to_stage_nd_list(strides, int(n_stages), ndim, "strides", 1)

        model_presets = {
            "S": {"exp_r": 2, "block_counts": [2] * 9, "checkpoint_style": None},
            "B": {"exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2], "block_counts": [2] * 9, "checkpoint_style": None},
            "M": {
                "exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2],
                "block_counts": [3, 4, 4, 4, 4, 4, 4, 4, 3],
                "checkpoint_style": "outside_block",
            },
            "L": {
                "exp_r": [3, 4, 8, 8, 8, 8, 8, 4, 3],
                "block_counts": [3, 4, 8, 8, 8, 8, 8, 4, 3],
                "checkpoint_style": "outside_block",
            },
        }

        preset = None
        if mednext_model_id is not None:
            preset = model_presets.get(str(mednext_model_id).upper())
            if preset is None:
                raise ValueError("mednext_model_id must be one of S/B/M/L")

        if exp_r is None:
            exp_r = preset["exp_r"] if preset is not None else 4

        if checkpoint_style is None and preset is not None:
            checkpoint_style = preset["checkpoint_style"]

        enc_counts, dec_counts = _resolve_block_counts(
            n_stages=int(n_stages),
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            block_counts=block_counts,
            preset_counts=preset["block_counts"] if preset is not None else None,
        )

        exp_full = _resolve_expansion(exp_r, int(n_stages))

        self.deep_supervision = bool(deep_supervision)
        self._expected_ds_outputs = max(int(n_stages) - 1, 1)

        self.core = MedNeXtCore(
            in_channels=int(input_channels),
            features_per_stage=[int(v) for v in features_per_stage],
            n_classes=int(num_classes),
            enc_kernel_sizes=enc_kernels,
            dec_kernel_sizes=dec_kernels,
            strides=stride_list,
            enc_block_counts=enc_counts,
            dec_block_counts=dec_counts,
            exp_r=exp_full,
            deep_supervision=self.deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            checkpoint_style=checkpoint_style,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        # decoder kernels are currently shared with encoder path blocks via stage kernels.
        # keep compatibility by storing them for potential downstream introspection.
        self._enc_kernels = enc_kernels
        self._dec_kernels = dec_kernels

        self.encoder = _MedNeXtEncoderProxy(
            stem=self.core.stem,
            stages=self.core.enc_stages,
            down_blocks=self.core.down_blocks,
            output_channels=[int(v) for v in features_per_stage],
        )
        self.decoder = _DecoderProxy(self)

    def set_deep_supervision(self, enabled: bool) -> None:
        self.deep_supervision = bool(enabled)
        self.core.set_deep_supervision(self.deep_supervision)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        out = self.core(x)
        if self.deep_supervision:
            outputs = list(out) if isinstance(out, (list, tuple)) else [out]
            if len(outputs) > self._expected_ds_outputs:
                outputs = outputs[: self._expected_ds_outputs]
            return outputs
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


MedNeXt = MedNeXtV1

__all__ = ["MedNeXtV1", "MedNeXt"]
