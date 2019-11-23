import numpy as np
import dlib
import sys
import pickle

predictor_path = "../model/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "../model/dlib_face_recognition_resnet_model_v1.dat"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def extract_embeddings(f):
    img = dlib.load_rgb_image(f)
    # get bounding box for the face in image
    result = detector(img, 1)
    #For the assignment we are assuming that each image has only 1 face detected
    d=result[0]
    shape = sp(img, d)
    # Compute the 128D vector that describes the face in img
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    return(face_descriptor)

emb1 = extract_embeddings(sys.argv[1])
emb2 = extract_embeddings(sys.argv[2])
dist = np.linalg.norm(np.array(emb1) - np.array(emb2))

model = pickle.load(open('../model/logreg_dist.sav', 'rb'))

dist = dist.reshape(1,1)
print('Same face: '+ str(model.predict(dist)[0]))
print('Similarity score: ' + str(model.predict_proba(dist)[0][1]))