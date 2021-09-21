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
# 다른 모든 모델들 넣기
# Self Driving Car algorithms



def main():
    modelname = sys.argv[1]
    data_dir = sys.argv[2]
    
    epochs=50

    data = []
    for root,dirs,files in os.walk(data_dir,topdown=False) :
        for file_name in files:
            full_path = os.path.join(root,file_name)
            data.extend(np.load(full_path,allow_pickle=True))
    #cv2.imshow("frame",data[0][0])
    #cv2.waitKey(5000)
    data = np.array(data)
    images = np.array(list(data[:,0] / 255.0),dtype=np.float)
    labels = np.array(list(data[:,1]),dtype=np.int)
    
    

    # CNN
    if modelname=="CNN":
        model=cnn()
    
    # AlexNet
    if modelname=="AlexNet":
        model=alexnet()

    # if modelname=="Inceptionv3":
    #     model=inceptionv3()    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(images, labels, epochs=epochs,batch_size=1000,
                        validation_data=None)
    hfivename='./test_model_'+modelname+'_epochs_'+str(epochs)+ '.h5'
    model.save(hfivename)
    #cv2.destroyAllWindows()
if __name__=='__main__':
    main()
