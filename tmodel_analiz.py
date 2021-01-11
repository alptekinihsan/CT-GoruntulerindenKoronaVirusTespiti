# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.models import model_from_json


arqu_model = "covid_new_data1.h5" # Model Dosyası
arqu_model_json = "covid_new_data1.json" # Json Dosyası

dogru_y=[]
tahmin_y=[]
x = np.load('mod_xtest.npy')
y = np.load('mod_ytest.npy')
json_file = open(arqu_model_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('covid_new_data1.h5')

y_tahmin= loaded_model.predict(x)
yp = y_tahmin.tolist()
yt = y.tolist()
count = 0
for i in range(len(y)):
    yy = max(yp[i])
    yyt = max(yt[i])
    tahmin_y.append(yp[i].index(yy))
    dogru_y.append(yt[i].index(yyt))
    if(yp[i].index(yy)== yt[i].index(yyt)):
        count+=1
acc = (count/len(y))*100
np.save('dogru1__mod02', dogru_y)
np.save('tahmin1__mod02', tahmin_y)

print("Test Paketinde Doğruluk : "+str(acc)+"%")


# Karısıklık matrisini almak için numpy dizilerini kullandık
y_dogru = np.load('dogru1__mod02.npy')   # Doğru olan verileri kayıt ettik
y_tahmin = np.load('tahmin1__mod02.npy')  # Tahmin  verilerini kayıt ettik
cm = confusion_matrix(y_dogru, y_tahmin) # Karışıklık matrisine çekildi
express = ["Covid-19 ","Normal"]
matrixx='Karışıklık Matrisi'
print(cm)


import itertools
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(matrixx)
plt.colorbar()
tick_marks = np.arange(len(express))
plt.xticks(tick_marks, express, rotation=45)
plt.yticks(tick_marks, express)
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen Değer')
plt.show()

total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))