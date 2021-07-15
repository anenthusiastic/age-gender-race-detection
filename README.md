# age-gender-race-detection

I have created this project for my AI course.
This app is used for age, gender and race detection from video and photo.
Firstly, app is asking you whether you want to make detection from a youtube video or a photo
if you choose youtube video,you should enter a youtube url
if you choose photo, you should enter filepath of photo that exist in your computer

You can find the agenet and gendernet caffe models in this link :https://talhassner.github.io/home/publication/2015_CVPR 

I use UTKFace dataset for training my machine learning classifier (SVC) for predicting race.Because i didn't find any efficient model for race detection

In the beginnig you should run ML_Classifier method. This is for creating a pickle file that is include weights for race predicting model. And then you can run main method
