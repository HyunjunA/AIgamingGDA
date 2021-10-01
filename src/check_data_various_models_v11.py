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
from alexnet import alexnet
from inceptionv3 import inception_v3

# from sklearn.utils import class_weight
# import sklearn.utils.class_weight 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from time import time
import random
# 다른 모든 모델들 넣기
# Self Driving Car algorithms

def preprocess_data(data):

    data = np.array(data)
    images = np.array(list(data[:,0] / 255.0),dtype=np.float)
    labels = np.array(list(data[:,1]),dtype=np.int)
    labels = np.argmax(labels, axis=1)
    return images,labels

def main():
    modelname = sys.argv[1]
    # modelname="AlexNet"
    data_dir = sys.argv[2]
    # data_dir="C:/Users/User/Desktop/ai-gaming/AIgamingGDA/src/data"
    
    epochs = int(sys.argv[3])
    # epochs = 1
    batch_size = int(sys.argv[4])
    # num_files = 3
    num_files = int(sys.argv[5])

    class_weight = {0 : 0.32,
                     1 : 3.63,
                     2 : 1.56,
                     3: 1.78,
                     4: 5.16,
                     5: 3.836,
                     6: 8.0,
                     7: 6.0,
                     8: 0.26
                     }


    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []


    # CNN
    if modelname=="CNN":
        model=cnn()
    
    # AlexNet
    if modelname=="AlexNet":
        model=alexnet()
        


    pathdockersavedmodelinlocal="C:/Users/Jun/Documents/StudyingDocker/AIgamingGDA/"
    # pathdockersavedmodelindocker="/usr/src/app-name/intersavedmodel/"
    pathdockersavedmodelindocker="./intersavedmodel/"

    savedmodelfile="test_model_"+modelname+"_epochs"+"_"+str(epochs)+"_batchsize_"+str(batch_size)+".h5"
    savedmodelfile="test_model_AlexNet_epochs_10_batchsize_500_45data.h5"

    savedmodelpathindocker=pathdockersavedmodelindocker+savedmodelfile
    savedmodelpathinlocal=pathdockersavedmodelinlocal+savedmodelfile
    
    # in docker
    anspath=path.exists(savedmodelpathindocker)
    if anspath==True:
        print("Load saved model file before training")
        model=load_model(savedmodelpathindocker)
    if anspath==False:
        print("Cannot load saved model file before training")
    

    # if modelname=="Inceptionv3":
    #     model=inceptionv3()    

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # opt=tf.keras.optimizers.Adagrad(
    #     learning_rate=0.001,
    #     initial_accumulator_value=0.1,
    #     epsilon=1e-07,
    #     name="Adagrad",
    #     **kwargs
    # )
    # opt=tf.keras.optimizers.Adagrad(
    #     learning_rate=0.001,
    #     initial_accumulator_value=0.1,
    #     epsilon=1e-07,
    #     name="Adagrad"
    # )

    # opt=tf.keras.optimizers.Ftrl(
    #     learning_rate=0.001,
    #     learning_rate_power=-0.5,
    #     initial_accumulator_value=0.1,
    #     l1_regularization_strength=0.0,
    #     l2_regularization_strength=0.0,
    #     name="Ftrl",
    #     l2_shrinkage_regularization_strength=0.0,
    #     beta=0.0,
    #     **kwargs
    # )
    # opt=tf.keras.optimizers.Ftrl(
    #     learning_rate=0.0001,
    #     learning_rate_power=-0.5,
    #     initial_accumulator_value=0.1,
    #     l1_regularization_strength=0.0,
    #     l2_regularization_strength=0.0,
    #     name="Ftrl",
    #     l2_shrinkage_regularization_strength=0.0,
    #     beta=0.0
    # )

    # model.compile(optimizer='adam',
    #                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #                 metrics=['accuracy'])


    model.compile(optimizer=opt,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])


    

    
    checked_file_name_list=[]
   
    filescountunit=2
    checked_file_name_list_total=0
    # files = os.listdir(data_dir)


    # Load saved model if it exists
    # model=load_model("/usr/src/app-name/test_model_AlexNet_epochs_50_batchsize_500.h5")
    pathdockersavedmodelinlocal="C:/Users/Jun/Documents/StudyingDocker/AIgamingGDA/"
    pathdockersavedmodelindocker="/usr/src/app-name/"
    savedmodelfile="test_model_"+modelname+"_epochs"+"_"+str(epochs)+"_batchsize_"+str(batch_size)+".h5"
    savedmodelpathindocker=pathdockersavedmodelindocker+savedmodelfile
    savedmodelpathinlocal=pathdockersavedmodelinlocal+savedmodelfile

    # in docker
    anspath=path.exists(savedmodelpathindocker)
    if anspath==True:
        print("Found saved model file!")
        model=load_model(savedmodelpathindocker)
    if anspath==False:
        print("There is not the saved model file!")

    #Main Training Loop
    print('Starting training')
    for e in range(epochs):
        print(f'Epoch {e}:')
        print('---------------------------------------------------------------------------------------------------------')
        #Get list of all file numbers, shuffle them
        file_nums = list(range(1,num_files+1))
        random.shuffle(file_nums)
        i = 0
        #iterate through all data
        while i < len(file_nums):
            data = []

            # Load 5 files
            print(f'Training on files {file_nums[i:i+5]}')
            for file_num in file_nums[i:i+5]:
                if file_num==7:
                    continue
                file_path = os.path.join(data_dir,f"training_data-{file_num}.npy")
                file_data = np.load(file_path,allow_pickle=True)
                data.extend(file_data)
            #Split into train and test
            train_split = int(len(data)*0.8)
            train = data[:train_split]
            test = data[train_split:]
            i += 5
            """while len(files)!=0:
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
                        break"""

            # print("Here")
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
            #kfoldnum=sys.argv[6]
            #kfoldnum=int(kfoldnum)
            #kfold = KFold(n_splits=kfoldnum, shuffle=True)

            start = time()
            batch_start = 0
            batch_no = 1
            #Generate batches from train dataset
            while batch_start < len(train):
                print(f"Batch {batch_no}")
                batch_data = train[batch_start:batch_start+batch_size]
                batch_start += batch_size
                X_train,y_train = preprocess_data(batch_data)
                # train_metrics = model.train_on_batch(X_train, y_train,class_weight=class_weight,
                #                                     reset_metrics=False,return_dict=True)
                train_metrics = model.train_on_batch(X_train, y_train, reset_metrics=False,return_dict=True)
                print(train_metrics)
                batch_no += 1


            # Generate generalization metrics
                """scores = model.evaluate(images[test], labels[test],  verbose=0)
                print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])
                # Increase fold number
                fold_no = fold_no + 1"""

            # Eval after training on 5 files
            batch_data = []
            X_test,y_test = preprocess_data(test)
            test_metrics = model.test_on_batch(X_test,y_test,reset_metrics=False,return_dict=True)
            print('Training metrics after 5 files:')
            print(train_metrics)
            print('Test metrics after 5 files:')
            print(test_metrics)
            print('')
            print("Time per five npy files /n")
            print(time() - start)

            # Save model every 20 files
            if i % 20 == 0:
                print('Saving Model')
                hfivename='./test_model_'+modelname+'_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.h5'
                model.save(hfivename)

        # Save model at the end of each epoch
        print('Saving Model End Epoch')
        hfivename='./test_model_'+modelname+'_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.h5'
        model.save(hfivename)

    #print final metrics
    print('Training finished!')
    print('Final training metrics:')
    print(train_metrics)
    print('Final test metrics:')
    print(test_metrics)

    #Final save model
    hfivename='./test_model_'+modelname+'_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.h5'
    model.save(hfivename)
    print('Model Saved')

if __name__=='__main__':
    main()