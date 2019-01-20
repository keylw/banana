import numpy as np
import cv2 as cv2 
import sys 

import csv


class NN():
    data = []

    @staticmethod
    def predict(banana):
        nn_banana = NN.data[0]
        nn_distance = sys.maxsize
        for i, test_banana in enumerate(NN.data):
            dist = NN.distance(banana, test_banana[1])
            if dist < nn_distance : 
                nn_distance = dist
                nn_banana = test_banana
        return nn_banana[0]

    @staticmethod
    def distance(x,y):
        print(x,y)
        return np.abs(x[0] - y[0]) + np.abs(x[1] - y[1]) + np.abs(x[2] - y[2])

    @staticmethod
    def train(datafile = "test.csv"):
        with open(datafile, 'r') as myfile:
            csv_reader = csv.reader(myfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                d = (row[3],[row[0],row[1],row[2]])
                NN.data.append(d)
        

def banana_mask(image, banana):
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    
    mask = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(mask,[banana],0,255,-1)
    return mask

def extract_banana(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,230,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    banana = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [banana], 0, (0,255,0), 3)

    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return banana, image

def extract_rgb(image, banana):
    mask = banana_mask(image, banana)
    area = cv2.contourArea(banana)
    mean_val = cv2.mean(image, mask = mask)
    # cv2.imshow('banana mask',mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(mean_val)
    B = mean_val[0]
    G = mean_val[1]
    R = mean_val[2]
    return [R,G,B]


data = []
for i in range(1,7):
    im = cv2.imread('b{}.png'.format(i))
    b, _img = extract_banana(im)
    d = extract_rgb(im, b)
    d.append(i)
    data.append(d)

with open("test.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for d in data: 
        wr.writerow(d)



im = cv2.imread('banana2.jpg')
b, _img = extract_banana(im)
d = extract_rgb(im, b)

NN.train()
print(NN.predict(d))
cv2.imshow('image', _img)
cv2.waitKey(0)
cv2.destroyAllWindows()

