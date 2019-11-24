# Face Verification

## Final Approach (../code/face_verification.py)

Step 1: Using face recognition tool - dlib to maps an image to a 128 dimensional vector space. (Extract embeddings from both the images)

Step 2: Find Euclidean distance/norm between the embeddings of both the images

Step 3: Train a Logistic Regression which takes input the Euclidean distance as it's feature and predicts wether the images are similar or not. (The idea behind uisng a Logistic Regression is that the model can change the continuous score into probablity and identify a threshold above which we can say that both the images are similar)

Step 4: Find similarity score between 2 images using predict_proba() function of LogisticRegression package


## Other Approaches

Approach 1: (../code/facenet_model.ipynb)
1. Using FaceNet model present in keras: This model is trained using a siamese network and triplet loss as the loss function
2. Difference between the embeddings of each pair was calculated and all the 128 numbers were fed to logistic regression as input to predict wether a pair of images are similar

Approach 2: (../code/dlibs_model.ipynb)
1. Use the embeddings calculated in approach 1 and instead of using embeddings difference, use Euclidean Norm as the only input feature

Approach 3: (../code/dlibs_model.ipynb)
1. Using dlib library to create the face embedding for each image
2. Difference between the embeddings of each pair was calculated and all the 128 numbers were fed to logistic regression as input to predict wether a pair of images are similar(same as approach 1)


## Data Preprocessing:

Face from all the images were cropped using mtcnn model for calculating FaceNet embeddings and frontal_face_detector(from dlibs library) for calculating dlibs embeddings.
The cropped face pixels were normalized for the case of FaceNet embeddings


## Challenges:

Embedding generation was a time consuming task for FaceNet case. Therefore we only got embeddings for 20 positive and 20 negative pairs and trained the two different Logistic Regression models on only 40 pairs. 
We cannot rely on the results of model trained on ImageNet embeddings due to its small training size.


## Conclusion:

1. Getting embeddings from Facenet model is quiet slow and dlibs is very fast. 
2. Both the models show very high accuracy when logistic regression model is trained using Frobinious norm of difference in embeddings.
3. Dlib embeddings with frobinious norm logistic regression give us the best and fastest results


### Final script : face_verification.py
This script will print similarity score and whether the pair of images has same/different face.

Syntax: python3 face_verification.py 'path/to/image1' 'path/to/image2'
