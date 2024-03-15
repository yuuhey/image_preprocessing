import pandas as pd
import numpy as np
from tools.func import *
import mahotas
import time


def Calculate_measure(df, save_path='./dataset/', mea_df_name='unknown'):

    '''statistic measure'''
    start_time = time.time()  # 시작 시간 기록
    print("Statistical measure is being calculated...")

    # about grayscale
    entropy = []
    laplacian = []
    brightness = []
    B_sd = []
    c_moment = []
    contra = []
    img_size = []
    edge_intensity = []
    edge_num = []
    sift = []
    psnr = []
    centroid_l = []

    for i in range(len(df)):
        bbox = df.loc[i,'bbox']
        
        # bbox 정보에서 x, y, w, h 추출
        x, y, w, h = map(int, bbox)

        # 이미지 로드
        image_file_path = df.loc[i,'image_file']
        image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

        height, width= image.shape
        img_size_value = width * height

        # 이미지를 bbox에 맞게 자르기
        cropped_image = image[y:y+h, x:x+w]
        entropy_value = calculate_image_entropy_(cropped_image)
        laplacian_value = calculate_laplacian_variance_(cropped_image)
        brightness_value = np.mean(cropped_image)
        sd_value = np.std(cropped_image)
        c_moment_value = moment(cropped_image)
        contrast_value = calculate_contrast(cropped_image)
        edge_num_value, edge_value = edge(cropped_image)
        SIFT_count_value = SIFT_count(cropped_image)
        psnr_value = calculate_psnr(cropped_image)
        cx, cy = c_moment_value
        centroid_l_value = centroid_degree(w, h, cx, cy)

        entropy.append(entropy_value)
        laplacian.append(laplacian_value)
        brightness.append(brightness_value)
        B_sd.append(sd_value)
        c_moment.append(c_moment_value)
        contra.append(contrast_value)
        img_size.append(img_size_value)
        edge_intensity.append(edge_value)
        edge_num.append(edge_num_value)
        sift.append(SIFT_count_value)
        psnr.append(psnr_value)
        centroid_l.append(centroid_l_value)

    # about RGB color
    red = []
    green = []
    blue = []
    color_var = []

    for i in range(len(df)):
        bbox = df.loc[i,'bbox']
        
        # bbox 정보에서 x, y, w, h 추출
        x, y, w, h = map(int, bbox)

        # 이미지 로드
        image_file_path = df.loc[i,'image_file']
        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지를 bbox에 맞게 자르기
        cropped_image = image[y:y+h, x:x+w]
        
        # 이미지 배열에서 각 채널의 값 가져오기
        red_channel = cropped_image[:, :, 0]
        green_channel = cropped_image[:, :, 1]
        blue_channel = cropped_image[:, :, 2]

        # 각 채널별 평균과 분산 계산
        red_mean, red_variance = np.mean(red_channel), np.var(red_channel)
        green_mean, green_variance = np.mean(green_channel), np.var(green_channel)
        blue_mean, blue_variance = np.mean(blue_channel), np.var(blue_channel)
        variance = np.mean([red_variance,green_variance,blue_variance])

        red.append(red_mean)
        green.append(green_mean)
        blue.append(blue_mean)
        color_var.append(variance)

    # about HSV
    saturation = []

    for i in range(len(df)):
        bbox = df.loc[i,'bbox']
        
        # bbox 정보에서 x, y, w, h 추출
        x, y, w, h = map(int, bbox)

        # 이미지 로드
        image_file_path = df.loc[i,'image_file']
        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 이미지를 bbox에 맞게 자르기
        cropped_image = image[y:y+h, x:x+w]

        saturation_value = np.mean(cropped_image[:, :, 1])

        saturation.append(saturation_value)

    df['entropy'] = entropy
    df['laplacian'] = laplacian
    df['brightness'] = brightness
    df['B_sd'] = B_sd
    df['red'] = red
    df['blue'] = blue
    df['green'] = green
    df['color_var'] = color_var
    df['c_moment'] = c_moment
    df['contrast'] = contra
    df['img_size'] = img_size
    df['edge_intensity'] = edge_intensity
    df['edge_num'] = edge_num
    df['saturation'] = saturation
    df['sift'] = sift
    df['psnr'] = psnr
    df['centroid_l'] = centroid_l

    df[['c_moment_x', 'c_moment_y']] = pd.DataFrame(df['c_moment'].tolist(), index=df.index)
    df['bbox_prop'] = df['bbox_size']/df['img_size'].values
    df = df[['image_file'] + [col for col in df.columns if col != 'image_file']]


    ''' Haralick feature'''
    print("Haralick feature is being calculated...")
    image_file = []
    haralick = []
    for i in range(len(df)):
        bbox = df.loc[i,'bbox']
        
        # bbox 정보에서 x, y, w, h 추출
        x, y, w, h = map(int, bbox)

        # 이미지 로드
        image_file_path = df.loc[i,'image_file']
        image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

        # 이미지를 bbox에 맞게 자르기
        cropped_image = image[y:y+h, x:x+w]

        # Haralick 텍스처 특징 계산
        features = mahotas.features.haralick(cropped_image).mean(axis=0)
        haralick.append(features)
        image_file.append(image_file_path)

    haralick_df = pd.DataFrame(haralick, columns=[
    'ASM', 'Contrast', 'Correlation', 'Variance', 'IDM', 'Sum_Average',
    'Sum_Variance', 'Sum_Entropy', 'Entropy', 'Difference_Variance',
    'Difference_Entropy', 'IMC1', 'IMC2'
    ])

    end_time = time.time()  # 종료 시간 기록
    execution_time = end_time - start_time  # 실행 시간 계산
    
    tmp = pd.concat([df, haralick_df], axis=1)
    tmp = tmp.sort_values(by='score', ascending=False).reset_index(drop=True)
    numeric_cols = tmp.select_dtypes(include=['float64', 'int64']).columns
    numeric_df = tmp[numeric_cols]
    df = pd.concat([tmp[['image_file', 'bbox']], numeric_df], axis=1)
    print(df.columns.to_list())
    print(f"All measures calculated in {execution_time} seconds.")
    df.to_csv(save_path+mea_df_name+'.csv', index=False)