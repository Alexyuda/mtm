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
exposure_times = np.array([1/500, 1/30, 1/50, 1/24390], dtype=np.float32)
n_points_match_registration = 200
search_radius = 100


# registrate all images to the first image
img_anchor = Image.open(imgsFn[0])
registrated_images = [np.array(img_anchor).astype(np.uint8)[:, :, ::-1].copy()]
img_anchor_gray = img_anchor.convert('LA')
for i in range(1, len(imgsFn)):
    img = Image.open(imgsFn[i])
    img_gray = img.convert('LA')
    pnt_list_src, pnt_list_dst = find_mtm_matches(anchor=img_anchor_gray,
                                                  img=img_gray,
                                                  patch_size=patch_size,
                                                  alpha=alpha,
                                                  n_points_to_match=n_points_match_registration,
                                                  search_radius=search_radius)

    M, mask = utils.ransac_points(src_pnts=pnt_list_src, dst_pnts=pnt_list_dst, th=5)

    pnt_list_src_ransac = [m[0] for m in zip(pnt_list_src, mask) if m[1] == 1]
    pnt_list_dst_ransac = [m[0] for m in zip(pnt_list_dst, mask) if m[1] == 1]

    img_reg = utils.warp_image(img, np.linalg.inv(M))
    registrated_images.append(np.array(img_reg).astype(np.uint8)[:, :, ::-1].copy())

    utils.show_matches(src_img=img_anchor, dst_img=img, src_pnts=pnt_list_src_ransac, dst_pnts=pnt_list_dst_ransac,
                       title=f'Match between anchor image {imgsFn[0]} and {imgsFn[i]}')
    utils.show_blend_images(img1=img_anchor, img2=img, title=f'{imgsFn[0]} and {imgsFn[i]} before registration')
    utils.show_blend_images(img1=img_anchor, img2=img_reg, title=f'{imgsFn[0]} and {imgsFn[i]} after registration')


#fuse registrated images to HDR image

# Exposure fusion using Debevec
merge_debevec = cv2.createMergeDebevec()
hdr_debevec = merge_debevec.process(registrated_images, times=exposure_times.copy())

# Tonemap HDR image
tonemap1 = cv2.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())

# Exposure fusion using Mertens
merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(registrated_images)

# Convert datatype to 8-bit and save
res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

cv2.imwrite("./data/hdr_result_debevec.jpg", res_debevec_8bit)
cv2.imwrite("./data/hdr_result_mertens.jpg", res_mertens_8bit)

_, ax = plt.subplots(2)
ax[0].imshow(res_debevec_8bit[:, :, ::-1])
ax[0].set_title('HDR result debevec method')
ax[1].imshow(res_mertens_8bit[:, :, ::-1])
ax[1].set_title('HDR result mertens method')
plt.show()