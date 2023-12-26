## person_keypoints_val2017.json
: GT annotation 파일 (611004rows)
    - 'keypoints', 'image_id'(unique key아님; bbox 여러개 존재하는 이미지), 'bbox'(x,y,w,h) 존재

## bbox_result.json
: bbox 검출 결과에서 filtering 및 처리 거친 최종 bbox 결과 파일
    - image_file: 이미지 파일의 경로입니다.
    - center: 이미지에서 사람의 중심 좌표를 나타내는 키입니다. "ndarray" 키는 중심 좌표의 값이 들어 있는 배열을 나타냅니다.
    - scale: 
    - rotation: 사람의 자세를 나타내는 값으로, 여기서는 항상 0으로 설정되어 있습니다.
    - bbox: 바운딩 박스의 좌표를 나타내는 키로, [x, y, w, h] 형식입니다.
    - bbox_score: 바운딩 박스의 신뢰도(또는 점수)를 나타내는 값입니다.
    - dataset: 데이터셋의 이름을 나타내는 값으로, 여기서는 "coco"입니다.
    - joints_3d: 3D 조인트의 좌표를 나타내는 키입니다. "ndarray" 키는 3D 조인트의 좌표 값이 들어 있는 배열을 나타냅니다.
    - joints_3d_visible: 3D 조인트의 가시성을 나타내는 키입니다. "ndarray" 키는 3D 조인트의 가시성 값이 들어 있는 배열을 나타냅니다.
    - bbox_id: 바운딩 박스의 고유 식별자를 나타내는 값으로, 각 바운딩 박스에 대해 0부터 시작하여 증가하는 값입니다.


## GT_result.json
: GT bbox로부터 예측하는 중간과정(모델 내부)에서 얻은 결과 (6352 rows)
    - image_id', 'bbox_id', 'keypoints', 'center', 'scale', 'score', 'area'

## result_keypoints.json' 
: 모델의 포즈예측 결과 (6334rows)
    - 'image_id', 'keypoints', 'scale', 'score', 'center'


@ merge 방법 : GT_result.json과 result_keypoints.json을 center를 기준으로 병합(center 값,유형 똑같음) 
-> person_keypoints_val2017.json에서 bbox에서 center값 구해서 같은 이미지 image_id 내에서 가까운 center값에 merge