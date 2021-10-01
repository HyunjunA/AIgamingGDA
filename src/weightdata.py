import numpy as np
import pandas as pd

import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from keras.models import load_model

from constants import IMAGE_WIDTH,IMAGE_HEIGHT
import os,sys
import os.path
from os import path

from CNN import cnn
from alexnetv2 import alexnet
from inceptionv3 import inceptionv3

# from sklearn.utils import class_weight
# import sklearn.utils.class_weight 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
# 다른 모든 모델들 넣기
# Self Driving Car algorithms



def main():
    # modelname = sys.argv[1]
    modelname="AlexNet"
    # data_dir = sys.argv[2]
    data_dir="C:/Users/Jun/Documents/StudyingDocker/AIgamingGDA/src/data"
    
    # epochs = sys.argv[3]
    epochs = 10
    # batch_size = sys.argv[4]
    batch_size = 500

    epochs=int(epochs)
    batch_size = int(batch_size)

    data = []

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []


    # CNN
    if modelname=="CNN":
        model=cnn()
    
    # AlexNet
    if modelname=="AlexNet":
        model=alexnet()

    # if modelname=="Inceptionv3":
    #     model=inceptionv3()    


    
    checked_file_name_list=[]
    for root,dirs,files in os.walk(data_dir,topdown=False) :
        for file_name in files:
            full_path = os.path.join(root,file_name)
            data.extend(np.load(full_path,allow_pickle=True))
            checked_file_name_list.append(file_name)
            
            # if len(checked_file_name_list)==2:
            #     new_files=set(files).difference(set(checked_file_name_list))
            #     files=list(new_files)
            #     break
            
        # print("Here")

    # data = np.array(data)
    # images = np.array(list(data[:,0] / 255.0),dtype=np.float)
    labels = np.array(list(np.array(data)[:,1]),dtype=np.int)





    # Create a pd.series that represents the categorical class of each one-hot encoded row
    # y_classes = labels.idxmax(1, skipna=False)
    y_classes =np.argmax(labels, axis=1)

    from sklearn.preprocessing import LabelEncoder

    # Instantiate the label encoder
    le = LabelEncoder()

    # Fit the label encoder to our label series
    le.fit(list(y_classes))

    # Create integer based labels Series
    y_integers = le.transform(list(y_classes))

    # Create dict of labels : integer representation
    labels_and_integers = dict(zip(y_classes, y_integers))

    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    sample_weights = compute_sample_weight('balanced', y_integers)

    class_weights_dict = dict(zip(le.transform(list(le.classes_)), class_weights))

     





    y_integers = np.argmax(labels, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))

    print("hello")


if __name__=='__main__':
    main()
