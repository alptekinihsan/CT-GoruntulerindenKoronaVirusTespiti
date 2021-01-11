import cv2
import numpy as np
from matplotlib import pyplot as plt

img =cv2.imread("database/normal/1.jpg")
plt.imshow(img)
plt.show()
def sharpness(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container
    pass
img = cv2.cvtColor(cv2.imread("database/normal/1.jpg"), cv2.COLOR_BGR2GRAY)
plt.imshow(img)
plt.show()
img = sharpness(img)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
plt.imshow(img)
plt.show()


imgi =cv2.imread("database/covid/1.jpg")
plt.imshow(imgi)
plt.show()

imgi =cv2.imread("database/covid/1.jpg")
plt.imshow(imgi)
plt.show()
def sharpness2(imgi):
    container2 = np.copy(imgi)
    size2 = container2.shape
    for i in range(1, size2[0] - 1):
        for j in range(1, size2[1] - 1):
            gx = (imgi[i - 1][j - 1] + 2*imgi[i][j - 1] + imgi[i + 1][j - 1]) - (imgi[i - 1][j + 1] + 2*imgi[i][j + 1] + imgi[i + 1][j + 1])
            gy = (imgi[i - 1][j - 1] + 2*imgi[i - 1][j] + imgi[i - 1][j + 1]) - (imgi[i + 1][j - 1] + 2*imgi[i + 1][j] + imgi[i + 1][j + 1])
            container2[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container2
    pass
imgi = cv2.cvtColor(cv2.imread("database/covid/1.jpg"), cv2.COLOR_BGR2GRAY)
plt.imshow(imgi)
plt.show()
imgi = sharpness2(imgi)
imgi = cv2.cvtColor(imgi, cv2.COLOR_GRAY2RGB)
plt.imshow(imgi)
plt.show()
