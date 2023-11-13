import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def apply_ahe(image_path, output_folder, clip_limit=2.0, tile_grid_size=(8,8)):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # AHE 객체 생성
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # AHE 적용
    img_ahe = clahe.apply(img)

    # 결과 이미지 저장
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img_ahe)

    # 원본 이미지와 AHE가 적용된 이미지를 병렬로 출력
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_ahe, cmap='gray')
    plt.title('AHE Applied')
    plt.axis('off')

    plt.show()

# 사용 예
# image_path = 'val2017/000000237071.jpg'
# output_folder = 'output/'
# apply_ahe(image_path, output_folder, clip_limit=5.0)