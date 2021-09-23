import numpy as np
import cv2

# from constants import IMAGE_WIDTH,IMAGE_HEIGHT
import os,sys


from sklearn.utils import class_weight
from sklearn.model_selection import KFold
# 다른 모든 모델들 넣기
# Self Driving Car algorithms



def main():
    # modelname = sys.argv[1]
    data_dir = sys.argv[1]
    
    # epochs = sys.argv[3]
    # batch_size = sys.argv[4]

    # epochs=int(epochs)
    # batch_size = int(batch_size)

    data = []

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []
    
    for root,dirs,files in os.walk(data_dir,topdown=False) :
        for file_name in files:
            full_path = os.path.join(root,file_name)
            data.extend(np.load(full_path,allow_pickle=True))
            print(sys.getsizeof(data))
    #cv2.imshow("frame",data[0][0])
    #cv2.waitKey(5000)
    data = np.array(data)
    images = np.array(list(data[:,0] / 255.0),dtype=np.float)
    labels = np.array(list(data[:,1]),dtype=np.int)

    # To solve imbalanced data problem
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels),
                                                 labels)
    
    # Define the K-fold Cross Validator 
    kfold = KFold(n_splits=10, shuffle=True)

    
   
if __name__=='__main__':
    main()
