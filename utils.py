import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ransac_points(src_pnts, dst_pnts, th=5.0):
    keypoints_src = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in src_pnts]
    keypoints_dst = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in dst_pnts]

    src_pts = np.float32([m.pt for m in keypoints_src]).reshape(-1, 1, 2)
    dst_pts = np.float32([m.pt for m in keypoints_dst]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M, mask


def warp_image(img, M):
    img_cv = np.array(img)[:, :, 0]
    warped_image_cv = cv2.warpPerspective(img_cv, M, (img_cv.shape[1], img_cv.shape[0]))
    warped_image = Image.fromarray(warped_image_cv).convert('LA')
    return warped_image


def show_matches(src_img, dst_img, src_pnts, dst_pnts, title=None):
    src_img = src_img.convert('RGB')
    dst_img = dst_img.convert('RGB')
    keypoints_src = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in src_pnts]
    keypoints_dst = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in dst_pnts]
    matches = [cv2.DMatch(i, i, 1) for i in range(len(src_pnts))]
    src_img_cv = np.array(src_img)[:, :, 0]
    dst_img_cv = np.array(dst_img)[:, :, 0]
    match_img_cv = cv2.drawMatches(src_img_cv, keypoints_src, dst_img_cv, keypoints_dst, matches, None)
    match_img = Image.fromarray(match_img_cv)

    if title is not None:
        draw = ImageDraw.Draw(match_img)
        font = ImageFont.truetype("arial.ttf", 25)
        draw.text((30, 30), title, fill="red", font=font, align="left")
    match_img.show()

def show_blend_images(img1, img2, title=None):
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')
    alphaBlended = Image.blend(img1, img2, alpha=.5)
    if title is not None:
        draw = ImageDraw.Draw(alphaBlended)
        font = ImageFont.truetype("arial.ttf", 25)
        draw.text((30, 30), title, fill="red", font=font, align="left")
    alphaBlended.show()
