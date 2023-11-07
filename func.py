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
        plt.axis('off')
        plt.imshow(img, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.hist(img.ravel(), 256, [0, 256], color='r')
        plt.xlabel(txt)

        plt.show()