import numpy as np
import matplotlib.pyplot as plt
import cv2

# 이미지 밝기 히스토그램
def histo(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    if img is not None:
        sd = round(np.std(img),3)
        avg = round(np.mean(img),3)
        txt = "average : "+str(avg)+" sd : "+str(sd)
        hist, bins = np.histogram(img.ravel(), 256, [0, 256])

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.title(image_path.replace("val2017/", "").replace(".jpg", "").replace("0", "")) #
        plt.axis('off')
        plt.imshow(img, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.hist(img.ravel(), 256, [0, 256], color='r')
        plt.xlabel(txt)

        plt.show()


# 표준편차로 리스트 절반씩 분리
def seperation_by_sd(lst):
    n = len(lst)

    if n%2==1:
        low_sd = lst[:n//2+1]
        high_sd = lst[n//2+1:]
    else:
        low_sd = lst[:n//2]
        high_sd = lst[n//2:]
    return low_sd, high_sd


# 찾을 image_id 검색
def find_id(target_image_id, json_data):
    # image_id가 일치하는 항목 찾기
    matching_item = next((item for item in json_data if item["image_id"] == target_image_id), None)

    # 찾은 항목이 있으면 score를 출력, 없으면 "Not found" 출력
    if matching_item:
        print("Score for image_id {}: {}".format(target_image_id, matching_item["score"]))
    else:
        print("Image_id {} not found in the JSON data.".format(target_image_id))
    return

# json 데이터에서 score 검색
# json 데이터에서 리스트 내 id에 맞는 score를 찾아 반환  
def find_id_list(search_lst, json_data):
    image_id_to_find = []

    for item in search_lst:
        image_id_to_find.append(int(item.replace("val2017/", "").replace(".jpg", "")))

    # 결과를 저장할 딕셔너리 초기화
    result_dict = {}

    # 각 image_id에 대해 반복
    for image_id in image_id_to_find:
        # 주어진 image_id에 해당하는 항목 찾기
        found_item = next((item for item in json_data if item["image_id"] == image_id), None)

        # 해당하는 항목이 있으면 score를 추출하여 결과 딕셔너리에 추가
        if found_item:
            result_dict[image_id] = found_item["score"]
        else:
            result_dict[image_id] = None  # 해당하는 항목이 없을 경우 None으로 표시
    return result_dict


def calculate_laplacian_variance_(image):
    # 이미지 읽기 (흑백으로 변환)
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 라플라시안 필터 적용
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # 분산 계산
    variance_of_laplacian = laplacian.var()

    return variance_of_laplacian

# 엔트로피 계산
def calculate_image_entropy_(image):
    # image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

    # 이미지 히스토그램 계산
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0,256])

    # 확률 계산
    prob = hist / np.sum(hist)

    # 엔트로피 계산
    entropy = -np.sum(prob * np.log2(prob + 1e-10))  # 1e-10은 로그 계산 시 분모가 0이 되는 것을 방지하기 위한 작은 값

    return entropy


# 모멘트 계산
def moment(image):

    moments = cv2.moments(image)

    # 중심 모멘트
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    # Raw 모멘트
    m00 = moments['m00']
    m10 = moments['m10']
    m01 = moments['m01']
    return cx, cy

# 대비
def calculate_contrast(image):
    # 최대 밝기와 최소 밝기 계산
    I_max = np.max(image).astype(np.float32)
    I_min = np.min(image).astype(np.float32)

    # 전역 대비 계산
    contrast = (I_max - I_min) / (I_max + I_min + 1e-8) # 분모에 작은 값을 더하여 오버플로 방지
    return(contrast)

# edge
def edge(image, threshold1=30, threshold2=100):
    # 에지 감지 수행 (Canny 에지 감지 사용)
    edges = cv2.Canny(image, threshold1, threshold2)
    # 에지 픽셀의 강도 출력
    edge_intensity = np.sum(edges) / (edges.shape[0] * edges.shape[1])

    return np.sum(edges), edge_intensity

# SIFT 특징점
def SIFT_count(image):

    # SIFT 알고리즘을 사용한 특징점 추출
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return len(keypoints)