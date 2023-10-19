# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# source : https://github.com/milesial/Pytorch-UNet/tree/master/unet

import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

sys.path.append("scripts/examples/")
from common import load_images

import numpy as np
from typing import Iterable, Tuple


class CoordinateMapping:
    def apply(self, positions):
        raise NotImplementedError

    def reverse(self, positions):
        raise NotImplementedError

    def __add__(self, other):
        return SequentialCoordinateMapping((self, other))

    def __neg__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class LinearCoordinateMapping(CoordinateMapping):
    def __init__(self, scale=1.0, bias=0.0) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias

    def apply(self, positions):
        device = (
            positions.device if isinstance(positions, torch.torch.Tensor) else "cpu"
        )
        return positions * self.scale.to(device) + self.bias.to(device)

    def reverse(self, positions):
        device = (
            positions.device if isinstance(positions, torch.torch.Tensor) else "cpu"
        )
        return (positions - self.bias.to(device)) / self.scale.to(device)

    def __add__(self, other):
        if isinstance(other, LinearCoordinateMapping):
            return LinearCoordinateMapping(
                self.scale * other.scale,
                self.bias * other.scale + other.bias,
            )
        elif isinstance(other, Identity):
            return self
        return CoordinateMapping.__add__(self, other)

    def __neg__(self):
        return LinearCoordinateMapping(
            scale=1.0 / self.scale,
            bias=-self.bias / self.scale,
        )

    def __str__(self):
        return f"x <- {self.scale} x + {self.bias}"


class Conv2dCoordinateMapping(LinearCoordinateMapping):
    @staticmethod
    def from_conv_module(module):
        assert (
            isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.MaxPool2d)
            or isinstance(module, torch.nn.ConvTranspose2d)
        )
        if isinstance(module, torch.nn.ConvTranspose2d):
            return -Conv2dCoordinateMapping(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
            )
        return Conv2dCoordinateMapping(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
        )

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1) -> None:
        # TODO(Pierre) : Generalize later if necessary
        assert dilation == 1 or dilation == (1, 1)

        kernel_size = torch.tensor(kernel_size)
        stride = torch.tensor(stride)
        padding = torch.tensor(padding)

        output_coord_to_input_coord = LinearCoordinateMapping(
            scale=stride,
            bias=-0.5 * stride - padding + kernel_size / 2,
        )
        input_coord_to_output_coord = -output_coord_to_input_coord

        LinearCoordinateMapping.__init__(
            self,
            input_coord_to_output_coord.scale,
            input_coord_to_output_coord.bias,
        )


class Identity(CoordinateMapping):
    def apply(self, positions):
        return positions

    def reverse(self, positions):
        return positions

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __neg__(self):
        return self

    def __str__(self):
        return "x <- x"


class SequentialCoordinateMapping(CoordinateMapping):
    def __init__(self, mappings: Iterable[CoordinateMapping]) -> None:
        super().__init__()
        self.mappings = tuple(mappings)

    def apply(self, positions):
        for mapping in self.mappings:
            positions = mapping.apply(positions)
        return positions

    def reverse(self, positions):
        for mapping in reversed(self.mappings):
            positions = mapping.reverse(positions)
        return positions

    def __radd__(self, other):
        if isinstance(other, SequentialCoordinateMapping):
            return SequentialCoordinateMapping(other.mappings + self.mappings)
        return SequentialCoordinateMapping((other,) + self.mappings)

    def __neg__(self):
        return SequentialCoordinateMapping(reversed(self.mappings))

    def __str__(self):
        return " <- ".join(f"({str(mapping)})" for mapping in reversed(self.mappings))


class CoordinateMappingComposer:
    def __init__(self) -> None:
        self._mappings = {}
        self._arrows = set()

    def _set(self, id_from, id_to, mapping):
        if (id_to, id_from) in self._arrows:
            raise RuntimeError(f"the mapping '{id_to}' <- '{id_from}' already exist")

        m = self._mappings.setdefault(id_to, {})
        m[id_from] = mapping

        m = self._mappings.setdefault(id_from, {})
        m[id_to] = -mapping

        self._arrows.add((id_to, id_from))
        self._arrows.add((id_from, id_to))

    def set(self, id_from, id_to, mapping: CoordinateMapping):
        if not isinstance(mapping, CoordinateMapping):
            raise RuntimeError(
                f"the provided mapping should subclass `CoordinateMapping` to provide coordinate mapping between {id_from} and {id_to}"
            )

        for node_id in self._mappings.get(id_from, {}):
            self._set(node_id, id_to, self._mappings[id_from][node_id] + mapping)

        self._set(id_from, id_to, mapping)

    def get(self, id_from, id_to):
        return self._mappings[id_to][id_from]


class CoordinateMappingProvider:
    def mappings(self) -> Tuple[CoordinateMapping]:
        raise NotImplementedError


def function_coordinate_mapping_provider(mapping=None):
    mapping = Identity() if mapping is None else mapping

    def wrapper(fn):
        class AugFn(CoordinateMappingProvider):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, *args, **kwds):
                return fn(*args, **kwds)

            def mappings(self) -> Tuple[CoordinateMapping]:
                return mapping

        return AugFn()

    return wrapper


def mapping_from_torch_module(module) -> CoordinateMapping:
    if isinstance(module, CoordinateMappingProvider):
        return module.mappings()
    elif isinstance(module, torch.nn.Conv2d):
        return Conv2dCoordinateMapping.from_conv_module(module)
    elif isinstance(module, torch.nn.ConvTranspose2d):
        return Conv2dCoordinateMapping.from_conv_module(module)
    elif isinstance(module, torch.nn.modules.pooling.MaxPool2d):
        return Conv2dCoordinateMapping.from_conv_module(module)
    elif isinstance(module, torch.nn.Sequential):
        return sum((mapping_from_torch_module(mod) for mod in module), Identity())
    elif (
        isinstance(module, torch.nn.modules.activation.ReLU)
        or isinstance(module, torch.nn.modules.activation.LeakyReLU)
        or isinstance(module, torch.nn.Identity)
        or isinstance(module, torch.nn.BatchNorm2d)
        or isinstance(module, torch.nn.InstanceNorm2d)
        or isinstance(module, torch.nn.GroupNorm)
    ):
        return Identity()
    else:
        raise RuntimeError(
            f"cannot get the coordinate mappings of module of type {type(module)}"
        )




def fast_nms(
        image_probs: torch.Tensor,
        nms_dist: int = 4,
        max_iter: int = -1,
        min_value: float = 0.0,
        ):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_dist >= 0)

    def max_pool(x):
        return F.max_pool2d(x, kernel_size=nms_dist * 2 + 1, stride=1, padding=nms_dist)
    
    off_th = 16

    # image_probs = image_probs*2*1000
    max_mask = max_pool(image_probs)
    max_mask = torch.clamp((image_probs-max_mask)*off_th+1, 0, 1)

    # return max_mask*image_probs

    image_probs1 = max_mask*image_probs
    max_mask1 = max_pool(image_probs1)
    max_mask1 = torch.clamp((image_probs1-max_mask1)*off_th+1, 0, 1)

    return max_mask1*image_probs1

    # image_probs2 = max_mask1*image_probs1
    # max_mask2 = max_pool(image_probs2)
    # max_mask2 = torch.clamp((image_probs2-max_mask2)*100000+1, 0, 1)
    # return max_mask2*image_probs2

def prob_map_to_points_map(
    prob_map: torch.Tensor,
    nms_dist: int = 3,
):
    nms = fast_nms(prob_map, nms_dist=nms_dist)
    # remove added channel
    prob_map = nms.squeeze(1)

    return prob_map  # batch_output

def logits_to_prob(logits: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
    max_v = logits.max()
    min_v = logits.min()
    logits = (logits - min_v)/((max_v - min_v)/16) - 8
    prob = torch.sigmoid(logits)

    return prob

def vgg_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    use_batchnorm: bool = True,
    non_linearity: str = "relu",
    padding: int = 1,
) -> torch.nn.Module:
    """
    The VGG block for the model.
    This block contains a 2D convolution, a ReLU activation, and a
    2D batch normalization layer.
    Args:
        in_channels (int): the number of input channels to the Conv2d layer
        out_channels (int): the number of output channels
        kernel_size (int): the size of the kernel for the Conv2d layer
        use_batchnorm (bool): whether or not to include a batchnorm layer.
            Default is true (batchnorm will be used).
    Returns:
        vgg_blk (nn.Sequential): the vgg block layer of the model
    """

    if non_linearity == "relu":
        non_linearity = torch.nn.ReLU(inplace=True)
    else:
        raise NotImplementedError

    # the paper states that batchnorm is used after each convolution layer
    if use_batchnorm:
        vgg_blk = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            non_linearity,
            torch.nn.BatchNorm2d(out_channels),
        )
    # however, the official implementation does not include batchnorm
    else:
        vgg_blk = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            non_linearity,
        )

    return vgg_blk

class DetectorHead(torch.nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels: int = 128,
        lat_channels: int = 256,
        out_channels: int = 1,
        use_batchnorm: bool = True,
        padding: int = 1,
        detach: bool = False,
    ) -> None:
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}

        self._detach = detach

        self._detH1 = vgg_block(
            in_channels,
            lat_channels,
            3,
            use_batchnorm=use_batchnorm,
            padding=padding,
        )

        if use_batchnorm:
            # no relu (bc last layer) - option to have batchnorm or not
            self._detH2 = nn.Sequential(
                nn.Conv2d(lat_channels, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # if no batch norm
            self._detH2 = nn.Sequential(
                nn.Conv2d(lat_channels, out_channels, 1, padding=0),
            )

    def mappings(self):
        mapping = mapping_from_torch_module(self._detH1)
        mapping = mapping + mapping_from_torch_module(self._detH2)
        return mapping

    def forward(self, x: torch.Tensor):
        if self._detach:
            x = x.detach()

        x = self._detH1(x)
        x = self._detH2(x)
        return x

class DescriptorHead(torch.nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 256,
        use_batchnorm: bool = True,
        padding: int = 1,
    ) -> None:
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}

        # descriptor head (decoder)
        self._desH1 = vgg_block(
            in_channels,
            out_channels,
            3,
            use_batchnorm=use_batchnorm,
            padding=padding,
        )

        if use_batchnorm:
            # no relu (bc last layer) - option to have batchnorm or not
            self._desH2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # if no batch norm - note that normailzation is calculated later
            self._desH2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
            )

    def mappings(self):
        mapping = mapping_from_torch_module(self._desH1)
        mapping = mapping + mapping_from_torch_module(self._desH2)
        return mapping

    def forward(self, x: torch.Tensor):
        x = self._desH1(x)
        x = self._desH2(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module, CoordinateMappingProvider):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def mappings(self):
        return mapping_from_torch_module(self.conv)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class MultiConv(nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels,
        out_channels,
        length=1,
        mid_channels=None,
        padding=1,
        kernel=3,
    ):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.channels = [in_channels] + [mid_channels] * (length - 1) + [out_channels]
        self.multi_conv = nn.Sequential(
            *sum(
                [
                    [
                        nn.Conv2d(
                            self.channels[i],
                            self.channels[i + 1],
                            kernel_size=kernel,
                            padding=padding,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.channels[i + 1]),
                        nn.ReLU(inplace=True),
                    ]
                    for i in range(length)
                ],
                [],
            )
        )

    def mappings(self):
        return mapping_from_torch_module(self.multi_conv)

    def forward(self, x):
        return self.multi_conv(x)


class ParametricDown(nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels,
        out_channels,
        length=1,
        use_max_pooling=True,
        padding=1,
        kernel=3,
    ):
        super().__init__()

        downscale_layer = (
            nn.MaxPool2d(2)
            if use_max_pooling
            else nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
        )

        self.maxpool_conv = nn.Sequential(
            downscale_layer,
            MultiConv(
                in_channels,
                out_channels,
                length,
                padding=padding,
                kernel=kernel,
            ),
        )

    def mappings(self):
        return mapping_from_torch_module(self.maxpool_conv)

    def forward(self, x):
        return self.maxpool_conv(x)


class ParametricUp(nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels,
        out_channels,
        hor_channels=None,
        length=1,
        bilinear=True,
        padding=1,
        kernel=3,
        hor_mapping=None,
        below_mapping=None,
    ):
        super().__init__()

        assert padding in {0, 1}
        self.padding = padding
        self.hor_channels = in_channels // 2 if hor_channels is None else hor_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, self.hor_channels, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                self.hor_channels,
                kernel_size=2,
                stride=2,
            )

        self.conv = MultiConv(
            2 * self.hor_channels,
            out_channels,
            length,
            padding=padding,
            kernel=kernel,
        )

        self._calculate_pad(hor_mapping, below_mapping)

    def _calculate_pad(self, hor_mapping=None, below_mapping=None):
        if (hor_mapping is None) or (below_mapping is None):
            self.top_pad = None
            self.left_pad = None
            return

        up_mapping = below_mapping + self.mappings()
        if not (hor_mapping.scale == up_mapping.scale).all():
            raise RuntimeError(
                f"only layer of same scale can be combine in upsampling layer : {hor_mapping.scale} != {up_mapping.scale}"
            )

        top_pad = (hor_mapping.bias[0] - up_mapping.bias[0]).item()
        left_pad = (hor_mapping.bias[1] - up_mapping.bias[1]).item()

        assert top_pad >= 0
        assert left_pad >= 0
        assert float(int(top_pad)) == top_pad
        assert float(int(left_pad)) == left_pad

        self.top_pad = int(top_pad)
        self.left_pad = int(left_pad)

    def mappings(self):
        return mapping_from_torch_module(self.up) + mapping_from_torch_module(self.conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        dy = x2.shape[2] - x1.shape[2]
        dx = x2.shape[3] - x1.shape[3]

        if self.top_pad is None:
            pad_y_top = dy // 2
        else:
            pad_y_top = self.top_pad
        pad_y_bottom = dy - pad_y_top

        if self.left_pad is None:
            pad_x_left = dx // 2
        else:
            pad_x_left = self.left_pad
        pad_x_right = dx - pad_x_left

        if self.padding:
            x1 = F.pad(x1, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom))
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        else:
            x2 = x2[..., pad_y_top:-pad_y_bottom, pad_x_left:-pad_x_right]

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class ParametricUNet(nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        input_feature_channels=64,
        n_scales=4,
        length=1,
        use_max_pooling=True,
        padding=1,
        kernel=3,
        up_channels=None,
        down_channels=None,
    ):
        nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_feature_channels = input_feature_channels
        self.down_channels = (
            [input_feature_channels * (2**i) for i in reversed(range(n_scales + 1))]
            if down_channels is None
            else [input_feature_channels] + down_channels
        )
        self.up_channels = (
            [input_feature_channels * (2**i) for i in range(n_scales + 1)]
            if up_channels is None
            else [self.down_channels[-1]] + up_channels
        )

        assert len(self.up_channels) == n_scales + 1
        assert len(self.down_channels) == n_scales + 1

        self.padding = padding
        self.length = length
        self.n_scales = n_scales
        self.kernel = kernel

        self.up_mappings = [None] * (n_scales + 1)

        self.inc = MultiConv(
            n_channels,
            input_feature_channels,
            length,
            padding=padding,
            kernel=self.kernel,
        )

        self.down_mappings = [None] * (n_scales + 1)
        self.down_mappings[0] = mapping_from_torch_module(self.inc)

        down = []
        for i in range(n_scales):
            layer = ParametricDown(
                self.down_channels[i],
                self.down_channels[i + 1],
                length,
                use_max_pooling=use_max_pooling,
                padding=padding,
                kernel=self.kernel,
            )

            down.append(layer)
            self.down_mappings[i + 1] = self.down_mappings[i] + mapping_from_torch_module(layer)

        up = []
        self.up_mappings[0] = self.down_mappings[-1]
        for i in range(n_scales):
            layer = ParametricUp(
                self.up_channels[i],
                self.up_channels[i + 1],
                self.down_channels[n_scales - 1 - i],
                length,
                bilinear=bilinear,
                padding=padding,
                kernel=self.kernel,
                hor_mapping=self.down_mappings[n_scales - i - 1],
                below_mapping=self.up_mappings[i],
            )

            up.append(layer)

            self.up_mappings[i + 1] = self.up_mappings[i] + mapping_from_torch_module(layer) 

        self.down = nn.ModuleList(down)
        self.up = nn.ModuleList(up)

        self.outc = OutConv(self.up_channels[-1], n_classes)

    def total_pad(self):
        pad = (1 - self.padding) * (self.kernel // 2)
        return (
            sum(2 * pad * (2**i) * self.length for i in range(self.n_scales))
            + pad * (2**self.n_scales) * self.length
        )

    def mappings(self):
        mapping = mapping_from_torch_module(self.inc)
        for down in self.down:
            mapping = mapping + mapping_from_torch_module(down)
        for up in self.up:
            mapping = mapping + mapping_from_torch_module(up)
        mapping = mapping + mapping_from_torch_module(self.outc)
        return mapping

    def forward(self, x):
        layers = [self.inc(x)]

        # downscale
        for down in self.down:
            layers.append(down(layers[-1]))

        # upscale
        x = layers.pop(-1)
        for up in self.up:
            x = up(x, layers.pop(-1))

        logits = self.outc(x)
        return logits



class SilkUnet(nn.Module):
    def __init__(
        self,
        in_channels,
        *,
        feat_channels: int = 128,
        lat_channels: int = 128,
        desc_channels: int = 128,
        use_batchnorm: bool = True,
        backbone=None,
        detector_head=None,
        descriptor_head=None,
    ):
        
        nn.Module.__init__(self)
        self.backbone = ParametricUNet(
            n_channels=1,
            n_classes=128,
            input_feature_channels=16,
            bilinear=False,
            use_max_pooling=True,
            n_scales=3,
            length=1,
            down_channels=[32, 64, 128],
            up_channels=[256, 256, 128],
            kernel=5,
            padding=0,
        )

        self.detector_head = (
            DetectorHead(
                in_channels=128,
                lat_channels=128,
                out_channels=1,
                use_batchnorm=True,
                padding=0,
            ))
        
        self.descriptor_head = (
            DescriptorHead(
                in_channels=128,
                out_channels=128,
                use_batchnorm=True,
                padding=0,
            )
        )

        self.descriptor_scale_factor = 1.41
        self.normalize_descriptors = True


    def forward(self, x: torch.Tensor):
        x1 = self.backbone(x)
        logits = self.detector_head(x1)
        descriptor = self.descriptor_head(x1)

        probs = logits_to_prob(logits, 1)
        prob_map = prob_map_to_points_map(probs)

        return prob_map, descriptor


if __name__ == "__main__":
    model_file = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-06-25/08-59-10/lightning_logs/version_0/checkpoints/epoch=9-step=70339.ckpt"
    model = SilkUnet(in_channels=1)
    
    state_dict    = torch.load(model_file)

    state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()}
    state_dict_new = {}
    for k, v in state_dict["state_dict"].items():
        if("_mods.model.backbone._backbone" in k):
            k = k.replace("_mods.model.backbone._backbone", "backbone")
            state_dict_new[k] = v
        if("_mods.model.backbone._heads._mods.logits" in k):
            k = k.replace("_mods.model.backbone._heads._mods.logits", "detector_head")
            state_dict_new[k] = v
        if("_mods.model.backbone._heads._mods.raw_descriptors" in k):
            k = k.replace("_mods.model.backbone._heads._mods.raw_descriptors", "descriptor_head")
            state_dict_new[k] = v
        

    model.load_state_dict(state_dict_new, strict=True)

    
    model = model.eval().to(device="cuda")


    # input_shape = [1,1,544,919]
    # input_data  = torch.rand(*input_shape)

    IMG_SHAPE = (544, 960)
    img_src = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/data/PingHu/XinXing1/snapshot_jtsj_1000_ckh8djx8z85c.jpg"
    images_q = cv2.imread(img_src)
    images_0 = load_images(img_src, img_shape=IMG_SHAPE)
    
    prob_map, descriptor = model(images_0)

    print("done")