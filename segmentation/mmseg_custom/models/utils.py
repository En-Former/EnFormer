import time
from typing import Tuple, Optional, Sequence, Union, Dict, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t, _ratio_2_t
from timm.layers import to_2tuple

__all__ = [
    'LayerNorm', 'Linear', 'ReparamDWConv', 'FFNx', 'BCHW2BHWC', 'BHWC2BCHW',
    'autopad', 'fuse_conv_bn', 'dilated2nondilated', 'merge_small_into_large_kernel',
    'compute_throughput', 'get_flops', 'get_flops_with_ptflops',
]


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = [normalized_shape, ]
        self.eps = eps
        self.data_format = data_format
        assert data_format in ["channels_last", "channels_first"]

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            return None


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 with_bn: bool = False, layer_scale: Optional[float] = None, deploy: bool = False,
                 input_to_channel_last: bool = False, output_to_channel_first: bool = False):
        super().__init__()
        self.deploy = deploy
        self.with_bn = with_bn
        self.bias = bias
        self.input_to_channel_last = input_to_channel_last
        self.output_to_channel_first = output_to_channel_first
        linear = []
        if self.input_to_channel_last:
            linear.append(BCHW2BHWC())
        linear.append(nn.Linear(in_features, out_features, bias=bias if not self.with_bn else False))
        if self.output_to_channel_first:
            linear.append(BHWC2BCHW())
        self.linear = nn.Sequential(*linear)
        if not self.deploy:
            if self.with_bn:
                if self.output_to_channel_first:
                    self.bn = nn.BatchNorm2d(out_features)
                else:
                    self.bn = nn.Sequential(
                        BHWC2BCHW(),
                        nn.BatchNorm2d(out_features),
                        BCHW2BHWC(),
                    )
            else:
                self.bn = None
        if (not self.deploy) and (layer_scale is not None):
            self.gamma = nn.Parameter(torch.ones(out_features) * layer_scale)
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.linear(x)

        x = self.linear(x)
        if self.with_bn:
            x = self.bn(x)
        if self.gamma is not None:
            if self.output_to_channel_first:
                x = self.gamma.unsqueeze(-1).unsqueeze(-1) * x
            else:
                x = self.gamma * x
        return x

    @torch.no_grad()
    def reparameterize(self):
        linear = self.linear[1] if self.input_to_channel_last else self.linear[0]
        if self.with_bn:
            bn = self.bn if self.output_to_channel_first else self.bn[1]
        else:
            bn = None
        linear_weight, linear_bias = fuse_conv_bn(linear, bn, self.gamma)

        new_linear = nn.Linear(linear.in_features, linear.out_features, bias=self.bias or self.with_bn)
        new_linear.weight.data = linear_weight
        if self.bias or self.with_bn:
            setattr(new_linear, 'bias', nn.Parameter(linear_bias))
        linear_layers = []
        if self.input_to_channel_last:
            linear_layers.append(BCHW2BHWC())
        linear_layers.append(new_linear)
        if self.output_to_channel_first:
            linear_layers.append(BHWC2BCHW())
        self.linear = nn.Sequential(*linear_layers)

        self.__delattr__('bn')
        self.__delattr__('gamma')
        self.deploy = True


class ReparamDWConv(nn.Module):
    kernel_settings = {
        # large_kernel_size: [small_kernel_sizes, dilated_rates],
        3: [[(1, 1)], [(1, 1)]],
        5: [[(3, 3), (3, 3)], [(1, 1), (2, 2)]],
        7: [[(5, 5), (3, 3), (3, 3)], [(1, 1), (2, 2), (3, 3)]],
    }

    def __init__(self, channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Optional[_size_2_t] = None, dilation: _size_2_t = 1, deploy: bool = False,
                 reparameterizable_kernel_settings: Dict[int, List] = None):
        super().__init__()
        if reparameterizable_kernel_settings is None:
            kernel_settings = self.kernel_settings
        else:
            kernel_settings = reparameterizable_kernel_settings

        self.deploy = deploy
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        if padding is None:
            self.padding = autopad(self.kernel_size, padding, self.dilation)
        else:
            self.padding = to_2tuple(padding)

        self.conv = nn.Conv2d(channels, channels, self.kernel_size, self.stride,
                              self.padding, self.dilation, groups=channels, bias=False)
        if not self.deploy:
            self.bn = nn.BatchNorm2d(channels)
            self.norm = nn.BatchNorm2d(channels)
            self.reparam_ks = kernel_settings[kernel_size][0]  # small kernel sizes
            self.reparam_dr = kernel_settings[kernel_size][1]  # dilated rates

            for i, (k, d) in enumerate(zip(self.reparam_ks, self.reparam_dr)):
                setattr(self, f'conv_{i}', nn.Conv2d(channels, channels, k, self.stride,
                                                     autopad(k, None, d), d, groups=channels, bias=False))
                setattr(self, f'bn_{i}', nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.conv(x)
        y = self.bn(self.conv(x))
        for i, _ in enumerate(self.reparam_ks):
            y = y + getattr(self, f'bn_{i}')(getattr(self, f'conv_{i}')(x))
        y = self.norm(y)
        return y

    @torch.no_grad()
    def reparameterize(self):
        conv_weight, conv_bias = fuse_conv_bn(self.conv, self.bn)
        for i, (k, d) in enumerate(zip(self.reparam_ks, self.reparam_dr)):
            small_conv = getattr(self, f'conv_{i}')
            small_bn = getattr(self, f'bn_{i}')
            small_conv_weight, small_kernel_bias = fuse_conv_bn(small_conv, small_bn)
            conv_bias += small_kernel_bias
            conv_weight = merge_small_into_large_kernel(conv_weight, small_conv_weight, self.dilation, d, None)

        self.conv.weight.data = conv_weight
        setattr(self.conv, 'bias', nn.Parameter(conv_bias))
        conv_weight_, conv_bias_ = fuse_conv_bn(self.conv, self.norm)
        self.conv.weight.data = conv_weight_
        setattr(self.conv, 'bias', nn.Parameter(conv_bias_))

        self.__delattr__('bn')
        self.__delattr__('norm')
        for i, _ in enumerate(self.reparam_ks):
            self.__delattr__(f'conv_{i}')
            self.__delattr__(f'bn_{i}')
        self.deploy = True


class FFNx(nn.Module):
    """Feed-Forward Network extension."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None, hidden_channels_scale: float = 4.0,
                 first_dw_kernel_size: Optional[_size_2_t] = None, hidden_dw_kernel_size: Optional[_size_2_t] = None,
                 hidden_act: Type[nn.Module] = nn.SiLU, drop_rate: _ratio_2_t = 0., layer_scale: Optional[float] = None,
                 deploy: bool = False):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)
        drop_rate = to_2tuple(drop_rate)
        self.deploy = deploy

        ffn_layers = []
        if first_dw_kernel_size is not None:
            ffn_layers.append(ReparamDWConv(in_channels, first_dw_kernel_size, 1,
                                            autopad(first_dw_kernel_size, None, 1), 1, deploy))
        ffn_layers.append(Linear(in_channels, hidden_channels, True, False, None, deploy,
                                 input_to_channel_last=True, output_to_channel_first=True if hidden_dw_kernel_size is not None else False))
        if hidden_dw_kernel_size is not None:
            ffn_layers.append(ReparamDWConv(hidden_channels, hidden_dw_kernel_size, 1,
                                            autopad(hidden_dw_kernel_size, None, 1), 1, deploy))
        if hidden_act is not None:
            import inspect
            signature = inspect.signature(hidden_act.__init__)
            if 'inplace' in signature.parameters:
                ffn_layers.append(hidden_act(inplace=True))
            else:
                ffn_layers.append(hidden_act())
        if drop_rate[0] != 0:
            ffn_layers.append(nn.Dropout(drop_rate[0]))
        ffn_layers.append(Linear(hidden_channels, out_channels, True, False, layer_scale, deploy,
                                 input_to_channel_last=True if hidden_dw_kernel_size is not None else False,
                                 output_to_channel_first=True))
        if drop_rate[1] != 0:
            ffn_layers.append(nn.Dropout(drop_rate[1]))
        self.ffn_layers = nn.Sequential(*ffn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_layers(x)

    @torch.no_grad()
    def reparameterize(self):
        for layer in self.ffn_layers:
            if hasattr(layer, 'reparameterize'):
                layer.reparameterize()
        self.deploy = True


# --------------------------------------------------- format reshape ---------------------------------------------------
class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor):
        return x.permute([0, 2, 3, 1])


class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor):
        return x.permute([0, 3, 1, 2])


# -------------------------------------------------------- utils -------------------------------------------------------
def autopad(kernel_size: Union[int, Sequence[int]], padding: Union[int, Sequence[int], None] = None,
            dilation: Union[int, Sequence[int]] = 1) -> Tuple[int, int]:
    """Auto padding conv2d function.

    Args:
        kernel_size (int or Sequence[int]): Convolutional kernel size.
        padding (optional, int or Sequence[int]): Padding size.
        dilation (int or Sequence[int]): Dilation size.

    Returns:
        Tuple[int, int]: Padding size.
    """

    if padding is not None:
        return padding

    if isinstance(kernel_size, int):
        k1, k2 = kernel_size, kernel_size
    elif isinstance(kernel_size, (tuple, list)):
        assert len(kernel_size) == 2  # avoid len(kernel_size) == 1 or len(kernel_size) > 2
        k1, k2 = kernel_size[0], kernel_size[1]
    else:
        raise ValueError(f'the type of kernel_size must be int, tuple or list, but got {type(kernel_size)}')
    assert k1 % 2 == 1 and k2 % 2 == 1, 'if use autopad, kernel size must be odd'

    if isinstance(dilation, int):
        d1, d2 = dilation, dilation
    elif isinstance(dilation, (tuple, list)):
        assert len(dilation) == 2  # avoid len(kernel_size) == 1 or len(kernel_size) > 2
        d1, d2 = dilation[0], dilation[1]
    else:
        raise ValueError(f'the type of dilation must be int, tuple or list, but got {type(dilation)}')

    if d1 > 1:
        k1 = d1 * (k1 - 1) + 1
    if d2 > 1:
        k2 = d2 * (k2 - 1) + 1

    return k1 // 2, k2 // 2  # padding


def fuse_conv_bn(
    conv: nn.Module,
    bn: Optional[nn.Module] = None,
    layer_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fuse Conv(Linear) and BN layer if BN exists, otherwise return original weight and bias.

    Args:
        conv (nn.Module): Conv/Linear layer.
        bn (Optional[nn.Module]): BN layer or None.
        layer_scale (Optional[torch.Tensor]): Layer scale factor. Default: None.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Fused weight and bias.
    """

    layer_scale = torch.tensor(1., device=conv.weight.device) if layer_scale is None else layer_scale.data.to(conv.weight.device)
    if len(conv.weight.shape) == 5:  # Conv3d
        reshape_format = (-1, 1, 1, 1, 1)
    elif len(conv.weight.shape) == 4:  # Conv2d
        reshape_format = (-1, 1, 1, 1)
    elif len(conv.weight.shape) == 3:  # Conv1d
        reshape_format = (-1, 1, 1)
    elif len(conv.weight.shape) == 2:  # Linear
        reshape_format = (-1, 1)
    else:
        raise ValueError(f'Unsupported weight shape: {conv.weight.shape}')

    if bn is None:
        weight = conv.weight * layer_scale.reshape(*reshape_format)
        if conv.bias is None:
            return weight, None
        else:
            return weight, conv.bias * layer_scale
    else:
        conv_bias = 0 if conv.bias is None else conv.bias
        std = (bn.running_var + bn.eps).sqrt()
        weight = conv.weight * (bn.weight / std * layer_scale).reshape(*reshape_format)
        bias = (bn.bias + (conv_bias - bn.running_mean) * bn.weight / std) * layer_scale
        return weight, bias


def dilated2nondilated(dilated_kernel: torch.Tensor, dilate_rate: _size_2_t = 1):
    """Convert dilated conv2d-weight to nondilated conv2d-weight, modified from
    `UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud,
    Time-Series and Image Recognition` (https://arxiv.org/abs/2311.15599).
    """

    dilate_rate = dilate_rate if isinstance(dilate_rate, tuple) else (dilate_rate, dilate_rate)
    if dilate_rate[0] == 1 and dilate_rate[1] == 1:
        return dilated_kernel

    identity_kernel = torch.ones((1, 1, 1, 1), dtype=dilated_kernel.dtype, device=dilated_kernel.device)
    if dilated_kernel.shape[1] == 1:  # kernel of DepthWise Conv
        nondilated_kernel = F.conv_transpose2d(dilated_kernel, identity_kernel, stride=dilate_rate)
    else:  # kernel of Dense or GroupWise Conv
        nondilated_kernel = torch.cat([
            F.conv_transpose2d(dilated_kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            for i in range(dilated_kernel.shape[1])
        ], dim=1)
    return nondilated_kernel


def merge_small_into_large_kernel(large_kernel: torch.Tensor, small_kernel: torch.Tensor,
                                  large_dilated_rate: _size_2_t = 1, small_dilated_rate: _size_2_t = 1,
                                  padding: Optional[Sequence[int]] = None):
    """Merge small kernel into large kernel."""

    large_kernel_sizes = large_kernel.shape[2:]
    small_kernel_sizes = small_kernel.shape[2:]
    large_dilated_rate = to_2tuple(large_dilated_rate)
    small_dilated_rate = to_2tuple(small_dilated_rate)

    assert small_dilated_rate[0] >= large_dilated_rate[0] and small_dilated_rate[1] >= large_dilated_rate[1], \
        f'The dilated rate {small_dilated_rate} of small kernel must be larger than or ' \
        f'equal to the dilated rate {large_dilated_rate} of large kernel.'
    assert small_dilated_rate[0] % large_dilated_rate[0] == 0 and small_dilated_rate[1] % large_dilated_rate[
        1] == 0, f'The dilated rate {small_dilated_rate} of small kernel must be divisible by the dilated rate ' \
                 f'{large_dilated_rate} of large kernel.'

    dilated_rate = (small_dilated_rate[0] // large_dilated_rate[0], small_dilated_rate[1] // large_dilated_rate[1])

    if large_kernel_sizes[0] == small_kernel_sizes[0] and large_kernel_sizes[1] == small_kernel_sizes[1]:
        if dilated_rate[0] != 1 or dilated_rate[1] != 1:
            raise ValueError('If small_kernel_sizes == large_kernel_sizes, dilated_rate must be same.')
        return large_kernel + small_kernel
    equivalent_kernel_sizes = tuple(d * (k - 1) + 1 for k, d in zip(small_kernel_sizes, dilated_rate))
    if equivalent_kernel_sizes[0] > large_kernel_sizes[0] or equivalent_kernel_sizes[1] > large_kernel_sizes[1]:
        raise ValueError('The equivalent kernel size is larger than the large kernel size.')
    equivalent_kernel = dilated2nondilated(small_kernel, dilated_rate)  # nondilated

    if padding is not None:
        assert len(padding) == 4, 'The length of padding must be 4, values: [padding_left, padding_right, ' \
                                  'padding_top, padding_bottom].'
        l, r, t, b = padding
        max_padding_h = large_kernel_sizes[0] - equivalent_kernel_sizes[0]
        max_padding_w = large_kernel_sizes[1] - equivalent_kernel_sizes[1]
        assert 0 <= l <= max_padding_w, f'padding_left must be in [0, {max_padding_w}], but got {l}'
        assert 0 <= r <= max_padding_w, f'padding_right must be in [0, {max_padding_w}], but got {r}'
        assert 0 <= t <= max_padding_h, f'padding_top must be in [0, {max_padding_h}], but got {t}'
        assert 0 <= b <= max_padding_h, f'padding_bottom must be in [0, {max_padding_h}], but got {b}'
    else:
        padding = [
            large_kernel_sizes[1] // 2 - equivalent_kernel_sizes[1] // 2,  # padding left
            large_kernel_sizes[1] // 2 - equivalent_kernel_sizes[1] // 2,  # padding right
            large_kernel_sizes[0] // 2 - equivalent_kernel_sizes[0] // 2,  # padding top
            large_kernel_sizes[0] // 2 - equivalent_kernel_sizes[0] // 2,  # padding bottom
        ]

    merged_kernel = large_kernel + F.pad(equivalent_kernel, padding, 'constant', 0)
    return merged_kernel


@torch.no_grad()
def compute_throughput(model, batch_size=256, resolution=224):
    torch.cuda.empty_cache()
    warmup_iters = 20
    num_iters = 100
    device = torch.device('cuda')

    model.eval()
    model.to(device)

    timing = []
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)

    # warmup
    for _ in range(warmup_iters):
        model(inputs)

    torch.cuda.synchronize()
    for _ in range(num_iters):
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)
    return (batch_size / timing.mean()).item()


def get_flops(model):
    # def fvcore_mul_flop_jit(inputs):
    #     from collections import Counter
    #     import numpy as np
    #     flop_dict = Counter()
    #     flop_dict["mul"] = np.prod(inputs[0].type().sizes())
    #     return flop_dict

    device = torch.device('cuda')
    model.eval()
    model.to(device)

    imgs = torch.rand(1, 3, 224, 224, device=device)
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, imgs)

    # flops.set_op_handle(**{'aten::mul': fvcore_mul_flop_jit,
    #                        'aten::div': fvcore_mul_flop_jit,
    #                        'aten::mul_': fvcore_mul_flop_jit,
    #                        'aten::add': fvcore_mul_flop_jit,
    #                        'aten::sum': fvcore_mul_flop_jit,
    #                        'aten::mean': fvcore_mul_flop_jit,
    #                        'aten::sub': fvcore_mul_flop_jit,
    #                        'aten::scatter_': fvcore_mul_flop_jit})
    print("FLOPs: ", flops.total() / 10. ** 9)


def get_flops_with_ptflops(model):
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        raise ImportError("Please install ptflops using the following command: pip install ptflops")
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
