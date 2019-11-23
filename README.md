# Face Verification

## Approach

Extract embedding an the image using one of the pretrained network and use that embeddings to find similarity score/detect similar face.
Used 2 different pretrained network to get embeddings in the form of 128D vector.
First is FaceNet model present in keras: This model is trained using a siamese network and triplet loss as the loss function
Second one is dlibs library

Pair of images were used to train the last leg of the model. Pair of images of same person was labeled 1 and pair of images of differet images was labelled as 0. The embeddings generated the pair of images were used to create two different Logistic regression model:
1. Frobinious norm of the difference in embeddings for each pair was calculated and was fed as input to the logistic regression model
2. Difference between the embeddings of each pair was calculated and all the 128 numbers were fed to logistic regression as input

Probability of image pair being same is used as the similarity score. This probility is derived using predict_proba() method of logistic regression model.


## Data Preprocessing:

Face from all the images were cropped using mtcnn model for calculating FaceNet embeddings and frontal_face_detector(from dlibs library) for calculating dlibs embeddings.
The cropped face pixels were normalized for the case of FaceNet embeddings


## Challenges:

Embedding generation was a time consuming task for FaceNet case. Therefore we only got embeddings for 20 positive and 20 negative pairs and trained the two different Logistic Regression models on only 40 pairs. 
We cannot rely on the results of model trained on ImageNet embeddings due to its small training size.


## Result:

Getting embeddings from Facenet model is quiet slow and dlibs is very fast. Both the models show very high accuracy when logistic regression model is trained using Frobinious norm of difference in embeddings.
Dlib embeddings with frobinious norm logistic regression give us the best and fastest results


Final script : face_verification.py
This script will print similarity score and whether the pair of images has same/different face.
Syntax: python3 face_verification.py 'path/to/image1' 'path/to/image2'
