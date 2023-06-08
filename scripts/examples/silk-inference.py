# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os, sys
sys.path.append("./")

from common import get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# IMAGE_0_PATH = "/datasets01/hpatches/01042022/v_adam/1.ppm"
# IMAGE_1_PATH = "/datasets01/hpatches/01042022/v_adam/2.ppm"



def main():
    root_path = "/workspace/mnt/storage/zhangjunkang/zjk1/data/sample_140_p1_pinhole/images"
    img_files = "140_lists.txt"
    img_lists = []
    with open(os.path.join(root_path, img_files)) as f:
        img_list_ = f.readlines()
        for img_ in img_list_:
            img_lists.append(img_.strip())

    # load model
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/assets/models/silk/pvgg-4.ckpt"
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-28/08-15-39/lightning_logs/version_0/checkpoints/epoch=9-step=39999.ckpt"
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-26/17-10-26/lightning_logs/version_0/checkpoints/epoch=5-step=4799.ckpt"
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-27/10-59-38/lightning_logs/version_0/checkpoints/epoch=5-step=4799.ckpt"
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-28/08-15-39/lightning_logs/version_0/checkpoints/epoch=3-step=15999.ckpt"
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-29/03-38-41/lightning_logs/version_0/checkpoints/epoch=6-step=4675.ckpt"
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-29/11-02-23/lightning_logs/version_0/checkpoints/epoch=0-step=2670.ckpt"
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-30/13-09-08/lightning_logs/version_0/checkpoints/epoch=7-step=25343.ckpt" # ***
    # ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-30/13-09-08/lightning_logs/version_0/checkpoints/epoch=8-step=28511.ckpt"
    ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-05-31/10-05-28/lightning_logs/version_0/checkpoints/epoch=9-step=31679.ckpt"
    
    model = get_model(checkpoint=ckpt, default_outputs=("sparse_positions", "sparse_descriptors"))

    IMAGE_0_PATH = "/workspace/mnt/storage/zhangjunkang/zjk1/data/sample_140_p1_pinhole/images/sample_preset1_ori.jpeg"
    # out_path = "./output/silk/vgg-punet_test0_e5_ratio0.5"
    # out_path = "./output/silk/vgg-punet_baseline_720x408_b8_1080p_lr0.0001_tem0.1_homo3_474x266_scale3_coco2WAndgaosu_e3_ratio0.6"
    # out_path = "./output/silk/vgg-punet_baseline_720x408_b8_1080p_lr0.0001_tem0.1_homo3_380x213_scale3_coco2WAndgaosu_e7_ratio1.0" # ***
    out_path = "./output/silk/vgg-punet_baseline_720x408_b8_1080p_lr0.00001_tem0.1_homo3_380x213_scale3_coco2WAndgaosu_e9_ratio1.0"
    IMG_SHAPE = (1080, 1920)
    # IMG_SHAPE = (540, 960)
    # IMG_SHAPE = (408, 720)
    if(not os.path.exists(out_path)):
        os.mkdir(out_path)
    images_0 = load_images(IMAGE_0_PATH, img_shape=IMG_SHAPE)
    sparse_positions_0_, sparse_descriptors_0 = model(images_0)
    for img_name in img_lists:
        # load image
        IMAGE_1_PATH = os.path.join(root_path, img_name)
        OUTPUT_IMAGE_PATH = os.path.join(out_path, os.path.basename(IMAGE_0_PATH)+"_"+os.path.basename(IMAGE_1_PATH)+".jpg") 
        images_1 = load_images(IMAGE_1_PATH, img_shape=IMG_SHAPE)

        # run model        
        sparse_positions_1, sparse_descriptors_1 = model(images_1)
        sparse_positions_0 = from_feature_coords_to_image_coords(model, sparse_positions_0_)
        sparse_positions_1 = from_feature_coords_to_image_coords(model, sparse_positions_1)

        # get matches
        matches = SILK_MATCHER(sparse_descriptors_0[0], sparse_descriptors_1[0])

        # create output image
        image_pair = create_img_pair_visual(
            IMAGE_0_PATH,
            IMAGE_1_PATH,
            IMG_SHAPE[0],
            IMG_SHAPE[1],
            sparse_positions_0[0][matches[:, 0]].detach().cpu().numpy(),
            sparse_positions_1[0][matches[:, 1]].detach().cpu().numpy(),
        )

        save_image(
            image_pair,
            os.path.dirname(OUTPUT_IMAGE_PATH),
            os.path.basename(OUTPUT_IMAGE_PATH),
        )

        print(f"result saved in {OUTPUT_IMAGE_PATH}")
    print("done")


if __name__ == "__main__":
    main()
