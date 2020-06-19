import numpy as np
from MTM import MTM_PWC
from PIL import Image
import numpy as np
import cv2


def find_mtm_matches(anchor, img, patch_size, alpha, n_points_to_match=50, search_radius=100):

    w, h = anchor.size

    n_points_1d = np.int(np.floor(np.sqrt(n_points_to_match)))
    scan_points_x = np.floor(np.linspace(np.max([search_radius, patch_size]), np.floor(w-np.max([search_radius, patch_size])), n_points_1d))
    scan_points_y = np.floor(np.linspace(np.max([search_radius, patch_size]), np.floor(h-np.max([search_radius, patch_size])), n_points_1d))

    pnt_list_src = []
    pnt_list_dst = []

    for scan_point_x in scan_points_x:
        for scan_point_y in scan_points_y:
            patch = anchor.crop((scan_point_x - patch_size/2, scan_point_y - patch_size/2,
                                 scan_point_x + patch_size/2, scan_point_y + patch_size/2))
            img_search_area = img.crop((scan_point_x - search_radius, scan_point_y - search_radius,
                                 scan_point_x + search_radius, scan_point_y + search_radius))
            D = MTM_PWC(img_search_area, patch, alpha)
            min_loc = np.where(D == np.min(D))
            pnt_list_src.append((scan_point_x, scan_point_y))
            pnt_list_dst.append((scan_point_x - search_radius + min_loc[1][0], scan_point_y - search_radius + min_loc[0][0]))

    return pnt_list_src, pnt_list_dst
