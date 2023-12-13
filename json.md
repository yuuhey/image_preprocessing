'person_keypoints_val2017.json' : GT annotation 파일 (611004rows)
    - 'keypoints', 'image_id'(unique key아님; bbox 여러개 존재하는 이미지), 'bbox'(x,y,w,h) 존재

'GT_result.json' : GT bbox로부터 예측하는 중간과정(모델 내부)에서 얻은 결과 (6352 rows)
    - image_id', 'bbox_id', 'keypoints', 'center', 'scale', 'score', 'area'

'result_keypoints.json' : 모델의 포즈예측 결과 (6334rows)
    - 'image_id', 'keypoints', 'scale', 'score', 'center'

merge 방법 : GT_result.json과 result_keypoints.json을 center를 기준으로 병합(center 값,유형 똑같음) 
-> person_keypoints_val2017.json에서 bbox에서 center값 구해서 같은 이미지 image_id 내에서 가까운 center값에 merge