# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os, sys
import cv2
import pickle
import numpy as np
sys.path.append("./")

from common import get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image

from sample_data_info import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():

    # load model
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/assets/models/silk/pvgg-4.ckpt"
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-30/13-09-08-asset/lightning_logs/version_0/checkpoints/epoch=7-step=25343.ckpt" # ***
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-30/13-09-08/lightning_logs/version_0/checkpoints/epoch=8-step=28511.ckpt"
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-31/10-05-28/lightning_logs/version_0/checkpoints/epoch=9-step=31679.ckpt" # lr0.00001
    ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-06-02/10-06-17/lightning_logs/version_0/checkpoints/epoch=0-step=3167.ckpt"  #nms=3
    
    model = get_model(checkpoint=ckpt, default_outputs=("sparse_positions", "sparse_descriptors"))


    root_path = "/workspace/mnt/storage/zhangjunkang/zjk1/data/sample_data-1"
    ips     = IPS
    presets = PRESETS
    poses   = POSES
    IMG_SHAPE = (1080, 1920)

    homo_graph_dict = {}
    
    for ip in ips:
        for preset in presets:
            preSetName = ip + "sample_preset" + preset + "_ori" + ".jpeg"
            preSetPath = os.path.join(root_path, ip, "sample_preset" + preset + "_ori" + ".jpeg")
            images_0 = load_images(preSetPath, img_shape=IMG_SHAPE)
            sparse_positions_0_, sparse_descriptors_0 = model(images_0)
            for pose in poses:
                for i in range(1, SAMPLE_NUM+1):
                    dst_img_name = ip + "sample_preset" + preset + "_" + pose + str(i) + ".jpeg"
                    dst_img_path = os.path.join(root_path, ip, "sample_preset" + preset + "_" + pose + str(i) + ".jpeg")
                    if(not os.path.exists(dst_img_path)):
                        continue
                    images_1 = load_images(dst_img_path, img_shape=IMG_SHAPE)
                    sparse_positions_1, sparse_descriptors_1 = model(images_1)
                    sparse_positions_0 = from_feature_coords_to_image_coords(model, sparse_positions_0_)
                    sparse_positions_1 = from_feature_coords_to_image_coords(model, sparse_positions_1)

                    # get matches
                    matches = SILK_MATCHER(sparse_descriptors_0[0], sparse_descriptors_1[0])

                    # create output image
                    sparse_positions_0 = sparse_positions_0[0][matches[:, 0]].detach().cpu().numpy()
                    sparse_positions_1 = sparse_positions_1[0][matches[:, 1]].detach().cpu().numpy()

                    sparse_positions_0_h = sparse_positions_0[:, [1,0]].astype(np.int32)
                    sparse_positions_1_h = sparse_positions_1[:, [1,0]].astype(np.int32)

                    H_, mask_ = cv2.findHomography(np.array(sparse_positions_0_h).squeeze(), np.array(sparse_positions_1_h).squeeze(), cv2.RANSAC, 3.0)
                    homo_graph_dict[preSetName+"_"+dst_img_name] = H_
                    # create output image
                    sparse_positions_0 = sparse_positions_0[:, [0,1]]
                    sparse_positions_1 = sparse_positions_1[:, [0,1]]
                    image_pair = create_img_pair_visual(
                        preSetPath,
                        dst_img_path,
                        None,
                        None,
                        sparse_positions_0,
                        sparse_positions_1,
                    )
                    save_image(image_pair, "./output/silk_eval/nms3-e0-0.6", preSetName+"_"+dst_img_name,)
    
    with open("silk-homograph-nms3-e0-0.6.pkl", "wb") as f:
        pickle.dump(homo_graph_dict, f)


if __name__ == "__main__":
    main()
