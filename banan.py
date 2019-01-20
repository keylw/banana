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
            dist = NN.euclidean_distance(banana, test_banana[1])
            if dist < nn_distance : 
                nn_distance = dist
                nn_banana = test_banana
        return nn_banana[0]

    @staticmethod
    def euclidean_distance(x,y):
        # print(x,y)    
        R_m = (x[0] - y[0])**2
        G_m = (x[0] - y[0])**2
        B_m = (x[0] - y[0])**2

        R_s = (x[0] - y[0])**2
        G_s = (x[0] - y[0])**2
        B_s = (x[0] - y[0])**2


        return np.sqrt(R_m + G_m + B_m + R_s + G_s + B_s) 

    @staticmethod
    def train(datafile = "training_banana/banana.csv"):
        with open(datafile, 'r') as myfile:
            csv_reader = csv.reader(myfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                print(row)
                d = (row[6],[row[0],row[1],row[2],row[3],row[4],row[5]])
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

    return banana, image

def extract_rgb_features(image, banana):
    mask = banana_mask(image, banana)
    area = cv2.contourArea(banana)
    mean, std = cv2.meanStdDev(image, mask = mask)

    B_mean = mean[0][0]
    G_mean = mean[1][0]
    R_mean = mean[2][0]

    B_std = std[0][0]
    G_std = std[1][0]
    R_std = std[2][0]

    print(R_mean)
    return [R_mean,G_mean,B_mean,R_std,G_std,B_std]

def predict(image):
    b, _img = extract_banana(image)
    d = extract_rgb_features(image, b)
    p = NN.predict(d)
    show_result(_img, p, d)


def train_NN():
    data = []
    for i in range(1,8):
        im = cv2.imread('training_banana/b{}.png'.format(i))
        b, _img = extract_banana(im)
        d = extract_rgb_features(im, b)
        d.append(i)
        data.append(d)    
    with open("training_banana/banana.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        for d in data: 
            wr.writerow(d)

def show_result(image, pred, rgb_values):
    import matplotlib.pyplot as plt
    
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


def main():
    NN.train()
    if len(sys.argv) < 2:
        print("missing argument: image")
        exit(-1)
    img = sys.argv[1]
    image = cv2.imread(img)
    predict(image)


if __name__ == "__main__":
    train_NN()
    main()

