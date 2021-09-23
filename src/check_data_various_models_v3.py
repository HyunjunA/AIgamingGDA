import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from constants import IMAGE_WIDTH,IMAGE_HEIGHT
import os,sys
from CNN import cnn
from alexnetv2 import alexnet
from inceptionv3 import inceptionv3

from sklearn.utils import class_weight
# import sklearn.utils.class_weight 
from sklearn.model_selection import KFold
# 다른 모든 모델들 넣기
# Self Driving Car algorithms



def main():
    modelname = sys.argv[1]
    data_dir = sys.argv[2]
    
    epochs = sys.argv[3]
    batch_size = sys.argv[4]

    epochs=int(epochs)
    batch_size = int(batch_size)

    data = []

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []
    
    for root,dirs,files in os.walk(data_dir,topdown=False) :
        for file_name in files:
            full_path = os.path.join(root,file_name)
            data.extend(np.load(full_path,allow_pickle=True))
    #cv2.imshow("frame",data[0][0])
    #cv2.waitKey(5000)
    data = np.array(data)
    images = np.array(list(data[:,0] / 255.0),dtype=np.float)
    labels = np.array(list(data[:,1]),dtype=np.int)


    # class_labels = [
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 1, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 1, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 1, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 1, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 1, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 1, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 1]
    # ]

    # To solve imbalanced data problem
    # class_weights = class_weight.compute_class_weight('balanced',
    #                                              np.unique(labels),
    #                                              labels)
    # class_weights = class_weight.compute_class_weight('balanced',
    #                                              class_labels,
    #                                              labels)
    sample_weights = class_weight.compute_sample_weight(class_weight,labels)
    # Define the K-fold Cross Validator 
    kfold = KFold(n_splits=10, shuffle=True)

    
    # CNN
    if modelname=="CNN":
        model=cnn()
    
    # AlexNet
    if modelname=="AlexNet":
        model=alexnet()

    # if modelname=="Inceptionv3":
    #     model=inceptionv3()    

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(images, labels):

        

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

        history = model.fit(images[train], labels[train], epochs=epochs,batch_size=batch_size,
                            validation_data=None, class_weight=class_weights)


        # Generate generalization metrics
        scores = model.evaluate(images[test], labels[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
                            
    hfivename='./test_model_'+modelname+'_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.h5'
    model.save(hfivename)
    #cv2.destroyAllWindows()
if __name__=='__main__':
    main()
