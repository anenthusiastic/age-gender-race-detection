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

I used oafy library and opencv methods to play video with youtube url. I used opencv library's methods for photo estimation.

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
