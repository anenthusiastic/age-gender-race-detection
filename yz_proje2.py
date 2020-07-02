import cv2
import pafy
import numpy as np
import pandas as pd
import os
import glob
from sklearn import svm
import face_recognition
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX
embedder = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7") #Yüzden 128d vektör şeklinde embedding çıkarmak için model

def load_caffe_models(): #Hazır aldığım  DL ile eğitilmiş modelleri yükleyen method
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')


    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

    return (age_net, gender_net)

def ML_classifier(): #Svm classifier ile yaş cinsiyet ve ırk tahminlemesi için eğittiğim method
    DATA_DIR = "UTKFace\\UTKFace"
    ID_RACE_MAP = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
    ID_GENDER_MAP = {0: 'Male', 1: 'Female'}

    def parse_filepath(filepath):
        try:
            path, filename = os.path.split(filepath)
            filename, ext = os.path.splitext(filename)
            age, gender, race, _ = filename.split("_")
            return int(age), ID_GENDER_MAP[int(gender)], ID_RACE_MAP[int(race)]
        except Exception as e:
            print(filepath)
            return None,None,None

    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    attributes = list(map(parse_filepath, files))
    df = pd.DataFrame(attributes)
    df.columns = ['age', 'gender', 'race']
    df['file'] = files
    df = df.dropna()
    df = df[(df['age'] > 10) & (df['age'] < 70)]
    df.index = np.arange(df.shape[0])
    row=df.shape[0]
    df = df.loc[0:row]
    df.index = np.arange(df.shape[0])
    df_features = pd.DataFrame(columns=np.arange(128))
    print(df.shape[0])
    for i in range(df.shape[0]):
        face_img=face_recognition.load_image_file(df['file'][i])
        faceBlob = cv2.dnn.blobFromImage(face_img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        feature_array = embedder.forward()


        if(len(feature_array)==1):
            df_features.loc[i] = feature_array[0]
        else:
            df.drop(i,inplace=True)
        print(i)

    X=df_features
    y=df[['race']]
    y=np.array(y)
    y.shape = (y.shape[0],)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)

    with open('race_classifier.pkl','wb') as file:
        pickle.dump(clf,file)

    racepred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, racepred))

    print("-----------------------------------------------")

    X = df_features
    y = df[['gender']]
    y = np.array(y)
    y.shape=(y.shape[0],)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)
    clf2 = svm.SVC(gamma='scale')
    clf2.fit(X_train, y_train)

    with open('gender_classifier.pkl','wb') as file:
        pickle.dump(clf2,file)

    genderpred = clf2.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, genderpred))


def video_detector(age_net,race_clf,gender_clf,url):
    # url of the video to predict Age and gender
    vPafy = pafy.new(url)
    play = vPafy.getbest(preftype="mp4")
    cap = cv2.VideoCapture(play.url)

    cap.set(3, 480)  # set width of the frame
    cap.set(4, 640)  # set height of the frame

    while True:
        ret, image = cap.read()

        faces = face_recognition.face_locations(image)
        if (len(faces) > 0):
            print("Found {} faces".format(str(len(faces))))
            for (top, right, bottom, left) in faces:
                cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 0), 2)
                # Get Face
                face_img = image[top:bottom, bottom-top:bottom-top+right-left].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1, ( 227, 227), MODEL_MEAN_VALUES, swapRB=False)

                # Cinsiyet tahminleme için aşağıdaki hazır modeli kullanıyordum fakat kötü accuracy yüzünden vazgeçtim
                '''gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                print("Gender : " + gender)'''
                # Predict Age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                print("Age Range: " + age)

                faceBlob = cv2.dnn.blobFromImage(face_img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                features = embedder.forward()

                race_predictionML = race_clf.predict(features)
                race = race_predictionML[0]
                print("Race : " + race)
                gender_predictionML = gender_clf.predict(features)
                gender = gender_predictionML[0]
                print("Gender : " + gender)

                overlay_text = "%s %s %s" % (gender, age,race)
                cv2.putText(image, overlay_text, (left, top), font, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', image)
        # 0xFF is a hexadecimal constant which is 11111111 in binary.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def photo_detector(age_net, race_clf,gender_clf,imagePath):

    image = face_recognition.load_image_file(imagePath)
    faces = face_recognition.face_locations(image)

    if (len(faces) > 0):
        print("Found {} faces".format(str(len(faces))))
        for (top, right, bottom, left) in faces:
            cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 0), 2)
            # Get Face
            face_img = image[top:bottom, bottom-top:bottom-top+right-left].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict Gender
            '''gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)'''

            # Yaş tahminleme için hazır dl modeli kullandım
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age)

            faceBlob = cv2.dnn.blobFromImage(face_img, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            features= embedder.forward()

            #Irk ve cinsiyet tahminleme için svm classifier kullandım
            race_predictionML = race_clf.predict(features)
            race = race_predictionML[0]
            print("Race : " + race)
            gender_predictionML = gender_clf.predict(features)
            gender=gender_predictionML[0]
            print("Gender : " + gender)


            overlay_text = "%s %s %s " % (gender, age, race)
            cv2.putText(image, overlay_text, (left, top), font, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('frame', RGB_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    print("Welcome to the age,gender and race prediction program!")
    print("Please select the process from menu")
    print("-------------------------------------")

    print("1.Make prediction from a Youtube video")
    print("2.Make prediction from image")
    print("3.Exit")

    choice = input("What is your choice:")
    while (choice not in ("1", "2", "3")):
        print("Not a valid option!")
        choice = input("What is your choice :")

    while(True):

        if(choice=="3"):
            return

        for i in range(3):
            print("Please wait while models loading...")

        age_net, gender_net = load_caffe_models()
        with open('race_classifier.pkl', 'rb') as file:
            race_model = pickle.load(file)
        with open('gender_classifier.pkl', 'rb') as file:
            gender_model = pickle.load(file)
        if (choice == "1"):
            for i in range(5):
                print("Dont forget to press 'q' for break the process!")

            url = input("Please enter a valid Youtube URL and leave space ( For example: https://www.youtube.com/watch?v=p9C1RGpOwVA ) :")
            video_detector(age_net, race_model,gender_model, url)
        else:
            path = input("Please enter a valid image path (For example: C\\Documents\\Images\\myimage) :")
            photo_detector(age_net,race_model,gender_model,path)


        print("---------------------------------------")
        print("1.Make prediction from a Youtube video")
        print("2.Make prediction from image")
        print("3.Exit")
        choice = input("What is your new choice :")
        while (choice not in ("1", "2", "3")):
            print("Not a valid option!")
            choice = input("What is your choice :")

main()