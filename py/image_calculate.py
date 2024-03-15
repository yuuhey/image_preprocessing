import argparse
import pandas as pd
import ast
import metric
import sys
import os

# current_dir = os.getcwd()
# project_root = os.path.abspath(os.path.join(current_dir, ".."))
# sys.path.append(project_root)

"""
usage example
python py/image_calculate.py --i /UHome/qtly_u/Predictive_Maintenance/image_preprocessing/image_processing/output_folder/ --df valid --name processedCLAHE
--i : image folder path to want to calculate measures
--df : image folder가 파생된 데이터의 종류 - valid인지 valid+train인지(addtrain)
--name : 저장할 데이터프레임의 이름 지정
"""

def get_arg():
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description='Calculate measure')

    # 인자 추가
    parser.add_argument('--i', type=str, help='Input image folder path')
    parser.add_argument('--df', type=str, help='Choose default dataframe; valid or addtrain')
    parser.add_argument('--name', type=str, help='calculated measure dataframe name')

    # 인자 파싱
    args = parser.parse_args()

    return args

def refine_dataframe(dataframe_path, image_folder):
    df = pd.read_csv(dataframe_path)
    df.drop(columns=['keypoints','area'], inplace=True)
    df["image_id"] = df["image_id"].astype(str).apply(lambda x: x.rjust(12, '0') + ".jpg")
    df['bbox'] = df['bbox'].apply(ast.literal_eval)
    df['image_file'] = str(image_folder)+df['image_id']
    df.drop(columns='image_id', inplace=True)
    return df


if __name__ == "__main__":
    args = get_arg()
    if args.df == 'valid':
        df_path = "/UHome/qtly_u/Predictive_Maintenance/image_preprocessing/dataset/merged_df.csv"
    elif args.df =='addtrain':
        df_path = "/UHome/qtly_u/Predictive_Maintenance/image_preprocessing/trainset_metric_calculate/train_merged_df.csv"
    else:
        print("No dataframe option provided. Please specify --df option.")
        exit()

    refined_df = refine_dataframe(df_path, args.i)
    metric.Calculate_measure(refined_df, mea_df_name=args.name)
    # metric.Calculate_measure(refined_df, mea_df_name=args.name)
    print("Completed, {}.csv has been saved in folder <dataset/>".format(args.name))