# CT-GoruntulerindenKoronaVirusTespiti
Derin Öğrenme ve Görüntü İşleme Tabanlı Korona Virüs Tespiti

Model 1090 tane CT taramalarından alınan görüntüler ile oluşturulmuştur.
Oluşturulan Model VGG16 Mimarisi Baz Alınarak Eğitim İşlemine Tabi Tutulmuştur.
Model Flask Framework'ü ile oluşturulan web uygulamasında kullanılmıştır.
Programın Kullanım Rehberi Çalıştırma.md' de belirtilmiştir.


"# Çalıştırmak İçin" 

Cmd ekranından klasörü attığınız dizinde çalıştırın

Cmd ekranında yada kullandığınız python düzenleyicilerindeki terminalde  şunu yazıyoruz  ::::: pip install -r kutuphaneler.txt  :::::::

Hazır Modeli Kullanmak İçin    python train.py   çalıştırıyoruz..

"#  HAZIR MODEL KULLANILMAK İSTENMEZ İSE SIRALI OLARAK ŞU İŞLEMLER YAPILMALIDIR.  "

Kütüphanelerin Eklenmesinden Sonra Şu İşlemler Yapılıyor.

"#  Model Oluşumu İçin   "

yeni_model.py   çalıştırıyoruz.

"#  Oluşturulan Modelin Analizi İçin    "

tmodel_analiz.py   çalıştırıyoruz.


"#  Oluşturulan Yeni Modelin Eğitimi İçin    " 

train.py    çalıştırıyoruz.

"#  Test İşlemi İçin    " 

test.py yada flask frameworkü ile yapılmış uygulamada oluşan modeli test ediyoruz.

"#  Test Edilen Görüntülerin Alınması İçin   " 

test_run.py   yada  Test-formatlı-resim.py 

çalıştırabilirsiniz..
