# age-gender-race-detection
I create this project for my AI introduction course
It is about face recognition (Spesificly age,gender and race detection)
This program ask for you want to make detection from a youtube video or a photo
if you choose youtube video,you should enter a youtube url
if you choose photo, you enter file path for photo that existing in the your computer

You can find the agenet and gendernet caffe models in this link :https://talhassner.github.io/home/publication/2015_CVPR 

I use UTKFace dataset for training my machine learning classifier (SVC) for predicting race.Because i didn't find any efficient model for race detection

In the beginnig you should run ML_Classifier method. This is for creating a pickle file for predicting race and then you can run main method
