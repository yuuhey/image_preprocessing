df.csv columns descriptions

'image_file' : 이미지 이름
'score’ : 모델 confidence
'width’ : bbox 너비
'height’ : bbox 높이
'bbox_size’ : bbox size = width*height
'entropy’ 
'laplacian’ : 선명도
'brightness’ : 범위로 계산한 밝기
'B_sd’ : 분산으로 계산한 밝기
'red’ : RGB 채널값
'blue’ : RGB 채널값
'green’ : RGB 채널값
'color_var’ : RGB값의 분산으로 계산한 채도
'contrast’ : 대비
'img_size’ : 이미지 크기
'edge_intensity’ : 엣지 강도
'saturation’ : HSV 채널의 채도
'sift’ : 특징점 개수
'c_moment_x’ : centroid moment x 값 (*centroid moment : 중심좌표를 기준으로 형태를 나타내는 지표)
'c_moment_y’ : centroid moment y 값
'bbox_prop’ : 이미지 면적 대비 바운딩박스가 차지하는 비율
--- 여기서부터는 haralick 계산으로 나온 지표 
'ASM’ : 에지 강도 정도
'Contrast’ : 인접한 픽셀값의 대비 
'Correlation’ : 이미지에서 한 방향의 밝기 변화가 다른 방향으로 얼마나 유사한지 (1에 가까울수록 변화가 일정)
‘Variance'
'IDM’ : Inverse Difference Moment (IDM): 픽셀 값의 변화가 얼마나 부드럽게 일어나는지를 나타내며, 값이 클수록 부드러운 텍스처를 의미
'Sum_Average'
'Sum_Variance'
'Sum_Entropy’ : 텍스처의 복잡성을 나타내며, 값이 클수록 텍스처가 복잡하다는 것을 의미
'Entropy’ : 이미지의 무질서도
'Difference_Variance’ : 픽셀 값 간의 차이의 분산
'Difference_Entropy’ : 픽셀 값 간의 차이의 무질서도
'IMC1’ (Information Measures of Correlation 1) : Correlation에 기반한 정보의 양
‘IMC2’ (Information Measures of Correlation 2) :  Correlation에 기반한 정보의 양