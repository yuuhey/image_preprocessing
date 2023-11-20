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


# 선명도 계산 - 라플라시안
def calculate_laplacian_variance(image_path):
    # 이미지 읽기 (흑백으로 변환)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 라플라시안 필터 적용
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # 분산 계산
    variance_of_laplacian = laplacian.var()

    return variance_of_laplacian


# 엔트로피 계산
def calculate_image_entropy(image_path):
    # 이미지 로드
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이미지 히스토그램 계산
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0,256])

    # 확률 계산
    prob = hist / np.sum(hist)

    # 엔트로피 계산
    entropy = -np.sum(prob * np.log2(prob + 1e-10))  # 1e-10은 로그 계산 시 분모가 0이 되는 것을 방지하기 위한 작은 값

    return entropy