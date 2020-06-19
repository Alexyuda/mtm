from PIL import Image, ImageDraw, ImageFont
from MTM import MTM_PWC
import numpy as np
from find_mtm_matches import find_mtm_matches
import cv2
from matplotlib import pyplot as plt
import utils


imgsFn = ['./data/hdr_1_500.jpg', './data/hdr_1_30.jpg', './data/hdr_1_50.jpg', './data/hdr_1_24390.jpg']
patch_size = 50
alpha = [0, 51, 102, 153, 204, 256]
n_points_match_registration = 200
search_radius = 100

registrated_image = [imgsFn[0]]

# registrate all images to the first image
img_anchor = Image.open(imgsFn[0]).convert('LA')
for i in range(1, len(imgsFn)):
    img = new_object = Image.open(imgsFn[i]).convert('LA')

    pnt_list_src, pnt_list_dst = find_mtm_matches(anchor=img_anchor,
                                                  img=img,
                                                  patch_size=patch_size,
                                                  alpha=alpha,
                                                  n_points_to_match=n_points_match_registration,
                                                  search_radius=search_radius)

    M, mask = utils.ransac_points(src_pnts=pnt_list_src, dst_pnts=pnt_list_dst, th=5)

    pnt_list_src_ransac = [m[0] for m in zip(pnt_list_src, mask) if m[1] == 1]
    pnt_list_dst_ransac = [m[0] for m in zip(pnt_list_dst, mask) if m[1] == 1]

    utils.show_matches(src_img=img_anchor, dst_img=img, src_pnts=pnt_list_src_ransac, dst_pnts=pnt_list_dst_ransac,
                       title=f'Match between anchor image {imgsFn[0]} and {imgsFn[i]}')

    img_reg = utils.warp_image(img, np.linalg.inv(M))
    utils.show_blend_images(img1=img_anchor, img2=img, title=f'{imgsFn[0]} and {imgsFn[i]} before registration')
    utils.show_blend_images(img1=img_anchor, img2=img_reg, title=f'{imgsFn[0]} and {imgsFn[i]} after registration')


