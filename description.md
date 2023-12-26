# 주요 json 파일
    GT - COCO annotation : person_keypoints_val2017.json
    model keypoints detection result - GT_result.json
    complete model keypoints detection result - result_keypoints.json


# GT에서 df 만들기 - 파일명(output.csv)
    make_merged_df_fromGT.ipynb(merged_df) -> calculate_metric_tomergedDF.ipynb(df, n_df)

    *df.csv : metric 계산 다된 완전한 df
    *n_df.csv : normalized_df


# file info
- mergedDF_visualization.ipynb : haralick 메트릭 제외 나머지 계산한 지표 시각화 


# additional info
- result_keypoints_image_processing.json랑 result_keypoints_small_base.json 같음
- bbox의 좌표 (x,y,w,h)