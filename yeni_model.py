
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from imutils import paths
import os

"""## Tensorflow/Keras Kütüphanelerinin Eklenmesi"""

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--database", default="database",
	help="path to input database")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid_new_data11.model",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

argument = {'database' : 'database','model' : 'covid_new_data11.model', 'plot' : 'plot.png"' }

print("Görüntüler Aktarılıyor ...")
imagePaths = list(paths.list_images(argument["database"]))
data = []
labels = []

# Görüntü uzantısı üzerinde döngü
for imagePath in imagePaths:
	# dosya adından sınıf etiketini çıkartımı
	label = imagePath.split(os.path.sep)[-2]
    

	# Görüntüyü yükleyip yeniden boyutlandırıyoruz.
	# 224x224 pixelin ayarlanması en,boy ihmal edildi.
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# Sırasıyla veri ve etiket listeleri güncellendi.
	data.append(image)
	labels.append(label)

# Pikseli ölçeklendirirken verileri ve etiketleri NumPy dizilerine dönüştürdük.
# [0, 255] arasındaki yoğunluğu aldık.
data = np.array(data) / 255.0
labels = np.array(labels)

# Label'larda ikili kodlma gerçekleştirdik.
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)   # Genişletme işlemi uygulandı (sebebi eğitilecek verideki başarımı arttırmak)

# Eğitilecek ve test edilecek verileri böldük

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=41)

print("Eğitim setindeki görüntü sayısı:", len(x_train))
print("Test setindeki görüntü sayısı:", len(x_test))
print("Doğrulama kümesindeki görüntü sayısı:", len(y_val))

np.save('mod_xtest', x_test)
np.save('mod_ytest', y_test)

num_features = 3   # özelikleri
num_classes = 2    #iki sınıf
width, height = 224, 224  # yükseklik ve genişlik
batch_size = 128  # 16 # aynı anda sinir ağında kaç veri geçecek
epochs = 1  # 2 # neural network 'ün üstünden kaç kere geçileceği belirlendi.

model = Sequential()  #Sıralı Model Kullanıldı.

"""## Katmanlar Modele Eklendi """

model.add(Conv2D(num_features, (3, 3), padding = 'same', kernel_initializer="normal",
                 input_shape = (width, height, 3)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(num_features, (3, 3), padding = "same", kernel_initializer="normal",
                 input_shape = (width, height, 3)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(2*num_features, kernel_initializer="normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(2*num_features, kernel_initializer="normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, kernel_initializer="normal"))
model.add(Activation("softmax"))

print(model.summary())


"""## Modelin Düzenlenmesi """

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), # Model optimize ediliyor.
              metrics=['accuracy'])

arq_model = "covid_new_data1.h5" #modeli kayıt etmek için
arq_model_json = "covid_new_data1.json"# json dosyasını kayıt etmek için
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
# Model eğitimi sırasında kayıp değerin istenilen basarımana ulaşıldı ise model eğitimini kesecek ancak başarım istenilen gibi değil ise belirtirlen sınırlamaya kadar çalışacak
checkpointer = ModelCheckpoint(arq_model, monitor='val_loss', verbose=1, save_best_only=True)
# Model eğitimi sırasında kayıp değerin kontrolu  yapılacak.
"""### Oluşturulacak Modeli JSON Dosyasına Kayıt Etmek"""
model_json = model.to_json()
with open(arq_model_json, "w") as json_file:
    json_file.write(model_json)
    
"""##Oluşturulan Modeli Eğitmek """
history = model.fit(np.array(x_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(x_val), np.array(y_val)),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, checkpointer])