import matplotlib.pyplot as plt


def visualize_keypoints(image_path, 
                        keypoints_list, 
                        skeleton=[[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]],
                        show=True):
    # 이미지를 읽어옵니다. 이미지 파일이 있는 경로로 바꿔주세요.
    
    img = plt.imread(image_path)

    # 이미지 크기 가져오기
    img_height, img_width, _ = img.shape

    plt.figure(figsize=(8, 8))
    plt.imshow(img)

    for keypoints in keypoints_list:

        # Keypoints를 x, y, score로 나누기
        x = keypoints[0::3]
        y = keypoints[1::3]
        score = keypoints[2::3]

        # 점 그리기
        for i, (x_coord, y_coord, s) in enumerate(zip(x, y, score)):
            plt.scatter(x_coord, y_coord, s=50, color='red', marker='o')
            plt.text(x_coord, y_coord, f'{s:.2f}', fontsize=8, color='white', ha='center', va='bottom')

        # 선 그리기
        for connection in skeleton:
            part1 = connection[0] - 1
            part2 = connection[1] - 1

            if 0 <= part1 < len(x) and 0 <= part2 < len(x):
                plt.plot([x[part1], x[part2]], [y[part1], y[part2]], color='blue')

    plt.axis('off')

    if show:
        plt.show()
