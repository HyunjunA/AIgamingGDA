import random
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from constants import IMAGE_WIDTH,IMAGE_HEIGHT,NUM_KEYS,W_VEC,A_VEC,S_VEC,D_VEC,WA_VEC,WD_VEC,SA_VEC,SD_VEC,NK_VEC,W_HEX,A_HEX,S_HEX,D_HEX
from keys import PressKey, ReleaseKey
from getkeys import key_check
from tensorflow import keras
import sys

W = 0
S = 1
A = 2
D = 3
WA = 4
WD = 5
SA = 6
SD = 7
NK = 8

def main():

    paused = False
    print("Starting in")
    for i in list(range(5))[::-1] :
        print(i)
        time.sleep(1)
    # load model
    model_path = sys.argv[1]
    network = keras.models.load_model(model_path)
    while True:
        start = time.time()
        if not paused:
            # Get screenshot
            img = grab_screen((0,0,800,600))
            img = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
            input_img = np.expand_dims(np.array(list(img / 255.0),dtype=np.float),0)
            # Get network prediction
            output_key = list(np.zeros((NUM_KEYS,),dtype=np.int))
            prediction = np.argmax(network.predict(input_img))
            output_key = prediction

            # Send output
            if output_key == W:
                PressKey(W_HEX)
                ReleaseKey(A_HEX)
                ReleaseKey(D_HEX)
                ReleaseKey(S_HEX)
                print('W')
            elif output_key == A:
                if random.randrange(0,3) == 1 :
                    PressKey(W_HEX)
                    print('WA')
                else :
                    ReleaseKey(W_HEX)
                    print('A')
                PressKey(A_HEX)
                ReleaseKey(W_HEX)
                ReleaseKey(D_HEX)
                ReleaseKey(S_HEX)

            elif output_key == S:

                PressKey(S_HEX)
                ReleaseKey(A_HEX)
                ReleaseKey(D_HEX)
                ReleaseKey(W_HEX)
                print('S')
            elif output_key == D:
                if random.randrange(0,3)  == 1 :
                    PressKey(W_HEX)
                    print('WD')
                else :
                    ReleaseKey(W_HEX)
                    print('D')
                PressKey(D_HEX)
                ReleaseKey(A_HEX)
                ReleaseKey(S_HEX)

            elif output_key == WA:
                PressKey(W_HEX)
                PressKey(A_HEX)
                ReleaseKey(S_HEX)
                ReleaseKey(D_HEX)
                print('WA')
            elif output_key == WD:
                PressKey(W_HEX)
                PressKey(D_HEX)
                ReleaseKey(S_HEX)
                ReleaseKey(A_HEX)
                print('WD')
            elif output_key == SA:
                PressKey(S_HEX)
                PressKey(A_HEX)
                ReleaseKey(W_HEX)
                ReleaseKey(D_HEX)
                print('SA')
            elif output_key == SD:
                PressKey(S_HEX)
                PressKey(D_HEX)
                ReleaseKey(W_HEX)
                ReleaseKey(A_HEX)
                print('SD')
            elif output_key == NK :
                if random.randrange(0,4) < 2:
                    PressKey(W_HEX)
                    print('W')
                else:
                    ReleaseKey(W_HEX)
                    print('NK')
                ReleaseKey(A_HEX)
                ReleaseKey(S_HEX)
                ReleaseKey(D_HEX)


        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)

            else:
                print('Pausing!')
                paused = True
                ReleaseKey(A_HEX)
                ReleaseKey(W_HEX)
                ReleaseKey(D_HEX)
                time.sleep(1)



if __name__=='__main__':
    main()