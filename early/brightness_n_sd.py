import numpy as np
import os
import cv2 

#각 이미지의 밝기와 표준편차 함수
def make_dict(folder_path = "val2017/"):
    # 폴더 내의 모든 파일 목록 가져오기
    file_list = os.listdir(folder_path)

    # 이미지 파일 확장자 (예: .jpg, .png)를 확인하여 이미지 파일만 선택
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    image_files = [f for f in file_list if os.path.splitext(f)[-1].lower() in image_extensions]

    # 전체 이미지의 밝기 평균과 표준편차 계산
    average = {}
    sd = {}

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        image = cv2.imread(image_path)
        average[image_path] = np.mean(image)
        sd[image_path] = np.std(image)

    average = sorted(average.items(), key= lambda item:item[1])
    sd = sorted(sd.items(), key= lambda item:item[1])
    return average, sd