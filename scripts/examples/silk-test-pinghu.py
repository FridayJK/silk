import os, sys
import cv2
import pickle
import numpy as np
import torch
import onnx
sys.path.append("./")

from common import get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image


def point_transform(H, pt):
    """
    @param: H is homography matrix of dimension (3x3) 
    @param: pt is the (x, y) point to be transformed
    
    Return:
            returns a transformed point ptrans = H*pt.
    """
    a = H[0,0]*pt[0] + H[0,1]*pt[1] + H[0,2]
    b = H[1,0]*pt[0] + H[1,1]*pt[1] + H[1,2]
    c = H[2,0]*pt[0] + H[2,1]*pt[1] + H[2,2]
    return [round(a/c), round(b/c)]

def homography_trans(H, pts):
    res_pts = []
    for pt in pts:
        x = pt[0]
        y = pt[1] 
        z = 1./(H[2][0]*x + H[2][1]*y + H[2][2])
        px = int((H[0][0]*x + H[0][1]*y + H[0][2])*z)
        py = int((H[1][0]*x + H[1][1]*y + H[1][2])*z)
        res_pts.append((px,py))
    return res_pts

def homography_trans2(H, pts):
    res_pts = []
    for pt in pts:
        x, y = point_transform(H, pt)
        res_pts.append((x,y))
    return res_pts


ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-06-25/08-59-10/lightning_logs/version_0/checkpoints/epoch=9-step=70339.ckpt"  # *** sota
# ckpt = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/var/silk-cli/run/training/2023-06-27/02-57-01/lightning_logs/version_0/checkpoints/epoch=5-step=42203.ckpt"  # *** sota

model = get_model(checkpoint=ckpt, default_outputs=("sparse_positions", "sparse_descriptors"))


root_path = "/workspace/mnt/storage/zhangjunkang/zjk_fileSystem/github/silk/data/PingHu"
# root_path = "/workspace/mnt/storage/zhangjunkang/zjk1/data/sample_data-1/172.168.137.236/"


# img_src = "XinHua/snapshot_jtsj_1000_ckh83i1bk0sg.jpg"
# img_dst = "XinHua/YHyuHIHuA37dgNBY7fWK3M1m_0_raw.jpg"
# img_src = "ZhaWang/snapshot_jtsj_1000_ckh89due3nk0.jpg"
# img_dst = "ZhaWang/87JfeLmcNCRyxl9TMdK2UiC9_0_raw.jpg"
img_src = "XinXing1/snapshot_jtsj_1000_ckh8djx8z85c.jpg"
img_dst = "XinXing1/8rIUCCrapusKpRPH_2iKKz4N_0_raw.jpg"
# img_src = "XinXing/snapshot_jtsj_1000_ckh8dm4vqozk.jpg"
# img_dst = "XinXing/8LLzpCoGWs_0nx9pShwFiKJR_0_raw.jpg"
# img_src = "XinHuaDangHu/snapshot_jtsj_1000_ckh838ybuj9c.jpg"
# img_dst = "XinHuaDangHu/dQG_SUm5yzZ0wlPYk2190il__0_raw.jpg"
# img_dst = "XinHuaDangHu/dAHt54mxMfbFea-i89drmZLQ_0_raw.jpg"
# img_src = "sample_preset1_ori.jpeg"
# # img_dst = "sample_preset1_ori.jpeg"
# img_dst = "sample_preset1_left4.jpeg"

img_src = os.path.join(root_path, img_src)
img_dst = os.path.join(root_path, img_dst)
# IMG_SHAPE = (1080, 1920)
IMG_SHAPE = (544, 960)



images_q = cv2.imread(img_src)
images_0 = load_images(img_src, img_shape=IMG_SHAPE)
# -----------------------onnx-----------------------
# onnx_file_name = "./test1.onnx"
# with torch.no_grad():
#     torch.onnx.export(model, images_0, onnx_file_name, opset_version=13, verbose=False, do_constant_folding=True,)

# onnx_model = onnx.load_model(onnx_file_name)
# onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
# onnx.checker.check_model(onnx_model)
# onnx.save(onnx_model, onnx_file_name)
# --------------------------------------------------


sparse_positions_0_, sparse_descriptors_0 = model(images_0)

images_r = cv2.imread(img_dst)
images_1 = load_images(img_dst, img_shape=IMG_SHAPE)
sparse_positions_1, sparse_descriptors_1 = model(images_1)
sparse_positions_0 = from_feature_coords_to_image_coords(model, sparse_positions_0_)
sparse_positions_1 = from_feature_coords_to_image_coords(model, sparse_positions_1)

# get matches
matches = SILK_MATCHER(sparse_descriptors_0[0], sparse_descriptors_1[0])

# create output image
sparse_positions_0 = sparse_positions_0[0][matches[:, 0]].detach().cpu().numpy()
sparse_positions_1 = sparse_positions_1[0][matches[:, 1]].detach().cpu().numpy()

sparse_positions_0_h = sparse_positions_0[:, [1,0]].astype(np.int32) * 2
sparse_positions_1_h = sparse_positions_1[:, [1,0]].astype(np.int32) * 2

H_, mask_ = cv2.findHomography(np.array(sparse_positions_0_h).squeeze(), np.array(sparse_positions_1_h).squeeze(), cv2.RANSAC, 3.0)

rect_label = os.path.join(os.path.dirname(img_src), "rect.txt")
rect = np.loadtxt(rect_label, delimiter=",").astype(np.int32)
# rect = np.array([[900, 500],[1000, 600]]).astype(np.int32)

pointDst_predict = homography_trans2(H_, rect)

cv2.rectangle(images_q, (rect[0,0], rect[0,1]), (rect[1,0], rect[1,1]), (0, 0, 255), 2)
cv2.rectangle(images_r, pointDst_predict[0], pointDst_predict[1], (0, 0, 255), 2)

stem0_name, stem1_name = os.path.basename(img_src), os.path.basename(img_dst)
out_path = "./output_pinghu_70339_nms2_mean_1filter"
if(not os.path.exists(out_path)):
    os.mkdir(out_path)
cv2.imwrite(os.path.join(out_path, stem0_name),images_q)
cv2.imwrite(os.path.join(out_path, stem1_name),images_r)