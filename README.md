TR

# Age, Gender and Race Detection (Yaş, Cinsiyet ve Irk Tespiti)

Bu proje, fotoğraflardan veya YouTube videolarından insan yüzlerini algılayarak **yaş, cinsiyet ve ırk** tahmininde bulunan Python tabanlı bir yapay zeka uygulamasıdır.

## 🎯 Proje Hakkında

Uygulama, verilen bir görsel girdi (resim dosyası veya YouTube video linki) üzerinde şu işlemleri gerçekleştirir:

1. Yüzleri tespit eder.
2. Tespit edilen yüzlerden öznitelik çıkarımı yapar.
3. Eğitilmiş Makine Öğrenmesi (SVM) ve Derin Öğrenme (Caffe) modellerini kullanarak kişinin demografik özelliklerini tahmin eder.

## 🚀 Özellikler

* **Çoklu Girdi Desteği:** Bilgisayarınızdaki bir fotoğrafı veya bir YouTube video bağlantısını analiz edebilir.
* **Karma Model Yapısı:**
* **Yaş:** Önceden eğitilmiş Caffe modeli (`age_net`).
* **Cinsiyet:** SVM Sınıflandırıcısı (Daha yüksek doğruluk için Caffe yerine tercih edilmiştir).
* **Irk:** UTKFace veri seti ile eğitilmiş SVM Sınıflandırıcısı.


* **Yüz Tespiti:** `face_recognition` kütüphanesi kullanılarak yüksek doğruluklu yüz bulma.

## 🛠 Kurulum ve Gereksinimler

Projenin çalışması için aşağıdaki Python kütüphanelerinin yüklü olması gerekmektedir.

### Gerekli Kütüphaneler

* OpenCV (`cv2`)
* NumPy
* Pandas
* face_recognition
* Pafy (YouTube videoları için)
* Scikit-learn (`sklearn`)
* Pickle

### Kurulum Adımları

1. **Depoyu Klonlayın:**
```bash
git clone https://github.com/anenthusiastic/age-gender-race-detection.git
cd age-gender-race-detection

```


2. **Bağımlılıkları Yükleyin:**
```bash
pip install opencv-python numpy pandas face-recognition pafy scikit-learn youtube-dl

```


*(Not: `pafy` ve `youtube-dl` sürümleri YouTube API değişikliklerine göre güncel olmalıdır.)*
3. **Model Dosyalarını İndirin:**
Proje, Caffe modellerine ihtiyaç duyar (`age_net.caffemodel`, `gender_net.caffemodel`). Bu dosyaları [bu bağlantıdan](https://talhassner.github.io/home/publication/2015_CVPR) indirip proje dizinine eklemeniz gerekebilir.

## 💻 Kullanım

Uygulamanın ana dosyası `yz_proje2.py`'dir.

1. **Modeli Eğitin (İlk Çalıştırma):**
Irk ve cinsiyet tahmini için kullanılan SVM modelinin ağırlıklarını oluşturmak adına, kod içerisindeki `ML_Classifier` fonksiyonunu bir kez çalıştırmanız gerekir. Bu işlem `pickle` dosyalarını oluşturacaktır.
2. **Uygulamayı Başlatın:**
```bash
python yz_proje2.py

```


3. **Girdi Seçimi:**
Program başladığında size soracaktır:
* **YouTube Videosu:** Bir YouTube linki girin.
* **Fotoğraf:** Bilgisayarınızdaki fotoğrafın dosya yolunu girin.



## 🧠 Nasıl Çalışır? (Teknik Detaylar)

1. **Veri Seti:** Irk tahmini modelini eğitmek için [UTKFace](https://www.kaggle.com/jangedoo/utkface-new) veri seti kullanılmıştır.
2. **Yüz Algılama:** Başlangıçta Haar Cascade denenmiş ancak başarısız olduğu için `face_recognition` kütüphanesine (HOG/CNN tabanlı) geçilmiştir.
3. **Öznitelik Çıkarımı:** Yüzlerin sayısal temsili (embedding) için `openface.nn4.small2.v1` modeli kullanılmıştır.
4. **Sınıflandırma:**
* Yaş tahmini için hazır CNN modeli kullanılmıştır.
* Cinsiyet ve Irk için öznitelikler çıkarıldıktan sonra SVM (Support Vector Machine) ile sınıflandırma yapılmıştır.

--------------------------------------------------------------------------------------------------------------------------------------------------

EN

# age-gender-race-detection

I have created this project for my AI course.
This app is used for age, gender and race detection from video and photo.
Firstly, app is asking you whether you want to make detection from a youtube video or a photo
if you choose youtube video,you should enter a youtube url
if you choose photo, you should enter filepath of photo that exist in your computer

You can find the agenet and gendernet caffe models in this link :https://talhassner.github.io/home/publication/2015_CVPR 

I used UTKFace dataset for training my machine learning classifier (SVC) for predicting race.Because i didn't find any efficient model for race detection

In the beginnig you should run ML_Classifier method. This is for creating a pickle file that is include weights for race predicting model. And then you can run main method


# 1) DESCRIPTION OF THE PROBLEM
As an application of facial recognition, I dealt with the problem of designing an application that performs age, gender and race detection using Machine Learning with high success rates.


# 2) PROBLEM SOLUTION STAGES
1. Finding the appropriate dataset for the application I want to develop
2.Face detection
3. Facial feature extraction
4. Classification for age, sex and race
5. Detection of age, gender and race from video and photo using OpenCV

# 2.1) Finding the Data
For the application I developed, I needed photos labeled in terms of age, gender and race. The best dataset I could find for this was the UTKFace dataset. I used it.
You can find the dataset in this link : https://www.kaggle.com/jangedoo/utkface-new
  
# 2.2) Face Detection

I made 2 different attempts for face detection. First, I used opencv's ready model for face detection (haarcascade). Later, I encountered the face that he could not detect many times and changed it. I used the face_locations method of the face_recognition library for face detection

# 2.3) Facial Features Extraction

I didn't need this at first because I used ready-made age and gender models, but then I couldn't find a successful ready-made model for race prediction, so I decided to do it with the SVM classifier. Then I had to do feature extraction and I used the ready model called openface.nn4.small2.v1. I used the face_encoding method of the face_recognition library before, but I found it unsuccessful.

# 2.4) Making classifications

He researched many different models. Initially, I used ready-made models named gender_net.caffemodel and age_net.caffemodel for age and gender estimation. Then I researched how to achieve more accurate results and stopped using the gender_net model and used the svm classifier. I continued to do the age estimation with the ready model.
I used SVM classifier for race prediction, same as gender.
 
# 2.5) Estimating from video and photo

I used pafy library and opencv methods to play video with youtube url. I used opencv library's methods for photo estimation.

# 3) SCREENS FROM APP

![image](https://user-images.githubusercontent.com/67736718/125955960-61fd8da6-de1b-48dc-9ef0-ca5ac6e47128.png)

![image](https://user-images.githubusercontent.com/67736718/125956003-fd0e7863-2695-4a46-b82b-2292f136b3fc.png)

![image](https://user-images.githubusercontent.com/67736718/125956031-a22f2200-7ee8-48d0-8039-426d13adf36a.png)


# 4) USED ENVIRONMENT, LIBRARIES AND SOURCES

# Used Libraries : 

1.OPENCV

2.NUMPY

3.PANDAS

4.FACE_RECOGNİTİON

5.PAFY

6.SKLEARN

7.PİCKLE

# Sources : 
1) https://www.sushanththarigopula.com/real-time-facial-recognition
2) https://towardsdatascience.com/predict-age-and-gender-using-convolutional-neural-network-and-opencv-fd90390e3ce6
3) https://www.youtube.com/watch?v=GT2UeN85BdA&t=32s
4) https://github.com/aakashjhawar/face-recognition-using-opencv
5) https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
6) https://www.kaggle.com/yhuan95/face-recognition-with-facenet
7) https://github.com/davidsandberg/facenet


## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir. Kullanılan veri setleri ve kütüphanelerin kendi lisans koşulları geçerlidir.
