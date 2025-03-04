# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy

import numpy as np
import skimage.io as io
from skimage.transform import rescale, resize
import torch

import torchvision
from silk.backbones.silk.silk import SiLKVGG as SiLK
from silk.backbones.superpoint.vgg import ParametricVGG

from silk.backbones.unet.unet import UNet
from silk.backbones.unet.unet import ParametricUNet

from silk.config.model import load_model_from_checkpoint
from silk.models.silk import matcher

# CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../../assets/models/silk/analysis/alpha/pvgg-4.ckpt")
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "../../assets/models/silk/coco-rgb-aug.ckpt"
)
DEVICE = "cuda:0"

# SILK_NMS = 0  # NMS radius, 0 = disabled
SILK_NMS = 3  # NMS radius, 0 = disabled
SILK_BORDER = 0  # remove detection on border, 0 = disabled
# SILK_THRESHOLD = 1.0  # keypoint score thresholding, if # of keypoints is less than provided top-k, then will add keypoints to reach top-k value, 1.0 = disabled
SILK_THRESHOLD = 0.6  # keypoint score thresholding, if # of keypoints is less than provided top-k, then will add keypoints to reach top-k value, 1.0 = disabled
# SILK_TOP_K = 10000  # minimum number of best keypoints to output, could be higher if threshold specified above has low value
SILK_TOP_K = 20000  # minimum number of best keypoints to output, could be higher if threshold specified above has low value
SILK_DEFAULT_OUTPUT = (  # outputs required when running the model
    "dense_positions",
    "normalized_descriptors",
    "probability",
)
SILK_SCALE_FACTOR = 1.41  # scaling of descriptor output, do not change
SILK_BACKBONE = ParametricVGG(
    use_max_pooling=False,
    padding=0,
    normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
)
SILK_BACKBONE_NNET = ParametricUNet(
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
# SILK_BACKBONE_NNET = ParametricUNet(
#     n_channels=1,
#     n_classes=128,
#     input_feature_channels=16,
#     bilinear=False,
#     use_max_pooling=True,
#     n_scales=4,
#     length=1,
#     down_channels=[32, 64, 128, 256],
#     up_channels=[512, 256, 256, 128],
#     kernel=5,
#     padding=1,
# )
# SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=1.0)
# SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=0.5)
# SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=0.6)
# SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=0.7)
SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=0.8)
# SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=0.9)
# SILK_MATCHER = matcher(postprocessing="double-softmax", threshold=1.0, temperature=0.1)
# SILK_MATCHER = matcher(postprocessing="double-softmax", threshold=0.95, temperature=0.1)
# SILK_MATCHER = matcher(postprocessing="double-softmax", threshold=0.9, temperature=0.1)
# SILK_MATCHER = matcher(postprocessing="double-softmax", threshold=0.6, temperature=0.1)
# SILK_MATCHER = matcher(postprocessing="none")


def load_images(*paths, img_shape=(1080, 1920), as_gray=True):
    img_tmps = [io.imread(path, as_gray=as_gray) for path in paths]
    images = [rescale(img_tmp, img_shape[0]/img_tmp.shape[0]) for img_tmp in img_tmps]


    # images = np.stack([io.imread(path, as_gray=as_gray) for path in paths])
    images = np.stack(images)
    images = torch.tensor(images, device=DEVICE, dtype=torch.float32)
    if not as_gray:
        images = images.permute(0, 3, 1, 2)
        images = images / 255.0
    else:
        images = images.unsqueeze(1)  # add channel dimension
    return images


def get_model(
    checkpoint=CHECKPOINT_PATH,
    nms=SILK_NMS,
    device=DEVICE,
    default_outputs=SILK_DEFAULT_OUTPUT,
):
    # load model
    model = SiLK(
        in_channels=1,
        backbone=deepcopy(SILK_BACKBONE_NNET),
        detection_threshold=SILK_THRESHOLD,
        detection_top_k=SILK_TOP_K,
        nms_dist=nms,
        border_dist=SILK_BORDER,
        default_outputs=default_outputs,
        descriptor_scale_factor=SILK_SCALE_FACTOR,
        padding=0,
    )
    model = load_model_from_checkpoint(
        model,
        checkpoint_path=checkpoint,
        state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
        device=device,
        freeze=True,
        eval=True,
    )
    return model
