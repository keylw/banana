import numpy as np
import cv2 as cv2 
import sys 
import _thread as thread


from model import NN
from model import Bayer
from model import MLP
import csv
import matplotlib.pyplot as plt
import banana_features as features


def predict(image):
    b, _img = features.extract_banana(image)
    d = features.extract_features(image, b)
    p = MLP.predict(d)
    show_result(_img, p, d)

def train():
    data = []
    images = []
    banana_contours = []

    for i in range(1,8):
        im = cv2.imread('training_banana/b{}.png'.format(i))
        b, _img = features.extract_banana(im)
        d = features.extract_features(im, b)
        d.append(i)
        data.append(d)

        # for plotting
        images.append(im)
        banana_contours.append(b)

    with open("training_banana/banana.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        for d in data: 
            wr.writerow(d)

def show_result(image, pred, rgb_values):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = [0,0,0,0,0,0,0]
    res[int(pred)-1] += 1

    label = ('1', '2', '3', '4', '5', '6', '7')
    y_pos = np.arange(len(label))

    plt.figure(1)
    plt.subplot(211)
    plt.imshow(image)

    plt.subplot(212)
    plt.bar(y_pos, res, align='center', alpha=0.5)
    plt.xticks(y_pos, label)
    plt.ylabel('Ripness level')
    plt.title('Banana')
    plt.show()

def from_cam():
    cap = cv2.VideoCapture(0)

    while True:
        r, frame = cap.read()
        if r:
            b, _img = features.extract_banana(frame)
            cv2.imshow("preview",_img)
            s_key = cv2.waitKey(1)
            if s_key & 0xFF == ord("s"): # start predict
                predict(frame)
                break
    

def main():
    MLP.train()
    if len(sys.argv) < 2:
        from_cam()        
        # print("missing argument: image")
        # exit(-1)
    elif(sys.argv[1] == 'train'):
        train()
    else:
        img = sys.argv[1]
        image = cv2.imread(img)
        predict(image)

if __name__ == "__main__":
    # train()
    main()  