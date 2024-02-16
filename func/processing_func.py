import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def apply_clahe_color(image_path, output_folder, clip_limit=2.0, tile_grid_size=(8, 8), brightness_factor=1.5):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 이미지 읽기
    img = cv2.imread(image_path)

    # 각 채널에 대해 밝기를 조절
    img_brightened = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

    # 각 채널에 대해 CLAHE 적용
    img_clahe = img_brightened.copy()
    for i in range(3):  # 0: B, 1: G, 2: R
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_clahe[:, :, i] = clahe.apply(img_brightened[:, :, i])

    # 결과 이미지 저장
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img_clahe)