
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--database", default="database",
	help="path to input database")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid_new_data1.h5",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Oluşturduğumuz Modeli (covid_new_data1.h5) h5 formatında ve Modelin (Loss, Accuracy , Trainin Loss ve Trainin Acc) değerlerini plot.png olarak aldık
argument = {'database' : 'database','model' : 'covid_new_data1.h5', 'plot' : 'plot.png"' }


INIT_LR = 1e-3
EPOCHS = 1 #7 # Neural Network' ün üstünden 7 defa geçirmek için katsayıyı belirledik.
BS = 16   # Kaç tane veri aynı anda Eğitilecek .

print("Görüntüler Aktarılıyor ...")
imagePaths = list(paths.list_images(argument["database"]))  # Verilerin yolunu listeye çektik.(Verileri çektik)

data = []
labels = []

# Listeye eşleştireceğimiz verileri böldük  .
for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]

# Verideki resimleri aldık  dönüştürdük ve boyutlarını belirledik.
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))    # Boyutlandırdık

# verileri yukarıda oluşturduğumuz dizilere ekleme işlemi yaptık.
	data.append(image)
	labels.append(label)

# Numpy dizilerine dönüştürme işlevi yaptık.
data = np.array(data) / 255.0
labels = np.array(labels)
# Label'da ikili kodlma gerçekleştirdik.
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)  # Genişletme işlemi uygulandı (sebebi eğitilecek verideki başarımı arttırmak)


# Eğitilecek verileri ve katmanları parçaladık % kaç test edileceğini belirledik
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.30, stratify=labels, random_state=42)



trainAug = ImageDataGenerator(  # Farklı resimleri üretecek
	rotation_range=15,   # resim çeşitliliği
	fill_mode="nearest")  # Doldurma Türü:: En yakın Komşu ya göre doldurulacak

#  Girdiyi VGG16 Modelinden Alıyoruz.
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# Modelleme işlemi yapılıyor Katmanlar Belirlendi
headModel = baseModel.output	# Base Modelin çıktısı Ana modele girdi olarak alındı
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)  # Ortalama havuzlama işlemi yapıldı
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)# Çıktıyı ana modelden alıyoruz.

# Modeli oluştururken oluşturduğumuz katmanları yeni eğitim ile güncelledik.
for layer in baseModel.layers:
	layer.trainable = False

print("Model Aktarılıyor ...")
# Eğitilecek verilere Optimize işlemi uygulandı
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)  #Var sayılan değerleri alındı.
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("Eğitiliyor ...")
# Modeli Eğitme işlemi uyguladık.
H = model.fit_generator(    # Modeli Eğitiyoruz Bu kısımda
	trainAug.flow(trainX, trainY, batch_size=BS),  # verileri Keras Kütüphanesinin İmageDataGeneratör metodu kullanarak  kayıt ediyoruz.
	steps_per_epoch=len(trainX) // BS,   # her epoch başına      47 görüntü geçirildi trainX için
	validation_data=(testX, testY),      # Doğrulama verileri
	validation_steps=len(testX) // BS,   # Her Epoch başına doğrulama için 47 görüntü geçirildi testX için.
	epochs=EPOCHS)

print("Aktarılıyor ...")
# İstenilen verileri çekmek için
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1) #iki katman olarak belirlendi
print(classification_report(testY.argmax(axis=1), predIdxs, #iki katman olarak belirlendi
						target_names=lb.classes_))


#Karışıklık Matrisini Oluşturduk.
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)  # Matrisi iki boyut olarak oldık
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total # Doğruluk
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) # Hassaslık
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) # Özgüllük

print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy  COVID-19")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Modeli diske kayıt etme
print("Modeli Kaydetme...")
model.save(args["model"], save_format="h5")