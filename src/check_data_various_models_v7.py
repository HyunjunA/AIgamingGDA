import numpy as np
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
from time import time
# 다른 모든 모델들 넣기
# Self Driving Car algorithms



def main():
    modelname = sys.argv[1]
    # modelname="AlexNet"
    data_dir = sys.argv[2]
    # data_dir="C:/Users/User/Desktop/ai-gaming/AIgamingGDA/src/data"
    
    epochs = sys.argv[3]
    # epochs = 1
    batch_size = sys.argv[4]
    # batch_size = 3

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
        


    model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])

    
    # if modelname=="Inceptionv3":
    #     model=inceptionv3()    


    
    checked_file_name_list=[]
   
    filescountunit=2
    checked_file_name_list_total=0
    # files = os.listdir(data_dir)


    for root_A,dirs_A,files_A in os.walk(data_dir,topdown=False) :
        root=root_A
        dirs=dirs_A
        files=files_A

    while len(files)!=0:
        for file_name in files:
            full_path = os.path.join(root,file_name)
            data.extend(np.load(full_path,allow_pickle=True))
            checked_file_name_list.append(file_name)
            
            if (len(checked_file_name_list)%filescountunit==0) or (len(files)<5 and len(files)==len(checked_file_name_list)):
                new_files=set(files).difference(set(checked_file_name_list))
                files=list(new_files)
                print("checked_file_name_list_len_total /n")
                checked_file_name_list_total+=len(checked_file_name_list)

                print(checked_file_name_list_total)
                checked_file_name_list=[]
                break
        
        # print("Here")

        data = np.array(data)
        images = np.array(list(data[:,0] / 255.0),dtype=np.float)
        labels = np.array(list(data[:,1]),dtype=np.int)



        # model=load_model("/usr/src/app-name/test_model_AlexNet_epochs_50_batchsize_500.h5")
        pathdockersavedmodelinlocal="C:/Users/Jun/Documents/StudyingDocker/AIgamingGDA/"
        pathdockersavedmodelindocker="/usr/src/app-name/"

        savedmodelfile="test_model_AlexNet_epochs"+"_"+str(epochs)+"_batchsize_"+str(batch_size)+".h5"

        savedmodelpathindocker=pathdockersavedmodelindocker+savedmodelfile
        savedmodelpathinlocal=pathdockersavedmodelinlocal+savedmodelfile
        
        # in docker
        anspath=path.exists(savedmodelpathindocker)
        if anspath==True:
            print("Found saved model file!")
            model=load_model(savedmodelpathindocker)
        if anspath==False:
            print("There is not the saved model file!")

        # in local
        # anspath=path.exists(savedmodelpathinlocal)
        # if anspath==True:
        #     model=load_model(savedmodelpathinlocal)
        # if anspath==False:
        #     print("There is not the saved model file!")



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
        # sample_weights = class_weight.compute_sample_weight(class_weight,labels)


        

        # y_integers = np.argmax(labels, axis=1)
        # class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        # d_class_weights = dict(enumerate(class_weights))

        # Define the K-fold Cross Validator 
        kfoldnum=sys.argv[5]
        kfoldnum=int(kfoldnum)
        kfold = KFold(n_splits=kfoldnum, shuffle=True)

        
        

        # K-fold Cross Validation model evaluation
        fold_no = 1

        start = time()

        for train, test in kfold.split(images, labels):

            


            history = model.fit(images[train], labels[train], epochs=epochs,batch_size=batch_size,
                                validation_data=None)




            # Generate generalization metrics
            scores = model.evaluate(images[test], labels[test],  verbose=0)
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            # Increase fold number
            fold_no = fold_no + 1
        
        print("Time per five npy files /n")
        print(time() - start)

        data=[]
        hfivename='./test_model_'+modelname+'_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.h5'
        model.save(hfivename)
                            
    hfivename='./test_model_'+modelname+'_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.h5'
    model.save(hfivename)
    print("Model Trained with entire files!")
    # data=[]
    #cv2.destroyAllWindows()
if __name__=='__main__':
    main()
