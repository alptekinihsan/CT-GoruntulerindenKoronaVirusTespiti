
import cv2 as  cv
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt


BS = 8
data = []

new_model = tf.keras.models.load_model('covid_new_data1.h5')  #Test Edilecek model yüklendi.
im= cv.imread('database/normal/1.jpg')               # Test edilecek  resim çekildi
imi = cv.imread('database/covid/1.jpg')              # Test edilecek  resim çekildi

image = cv.cvtColor(im, cv.COLOR_BGR2RGB)            # Normal
images = cv.cvtColor(imi, cv.COLOR_BGR2RGB)          # Covid
#image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
plt.imshow(image)
plt.show()
plt.imshow(images)
plt.show()

