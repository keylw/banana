import numpy as np
import cv2 as cv2 
import sys 

import csv
import matplotlib.pyplot as plt


class NN():
    training_data = []
    @staticmethod
    def predict(banana):
        label = -1
        nn_distance = sys.maxsize
        for i, test_banana in enumerate(NN.training_data):
            # dist = NN.rgb_euclidean_distance(banana, test_banana[1])
            dist = NN.hist_euclidean_distance(banana, test_banana[1])
            if dist < nn_distance : 
                nn_distance = dist
                label = test_banana[0]
        return label

    @staticmethod
    def rgb_euclidean_distance(x,y):
        # print(x,y)    
        R_m = (x[0] - y[0])**2
        G_m = (x[1] - y[1])**2
        B_m = (x[2] - y[2])**2

        R_s = (x[3] - y[3])**2
        G_s = (x[4] - y[4])**2
        B_s = (x[5] - y[5])**2

        return np.sqrt(R_m + G_m + B_m + R_s + G_s + B_s) 

    @staticmethod
    def hist_euclidean_distance(x,y):
        # print(x,y)    
        G_m = (x[0] - y[0])**2
        G_s = (x[1] - y[1])**2
        R_m = (x[2] - y[2])**2
        R_s = (x[3] - y[3])**2

        return np.sqrt(R_m + R_s + G_m + G_s ) 

    @staticmethod
    def train(datafile = "training_banana/banana.csv"):
        with open(datafile, 'r') as myfile:
            csv_reader = csv.reader(myfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                print(row)
                # d = (row[6],[row[0],row[1],row[2],row[3],row[4],row[5]])
                d = (row[4],[row[0],row[1],row[2],row[3]])

                NN.training_data.append(d)
        

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

def extract_histogram_features(image, banana):
    mask = banana_mask(image, banana)
    color = ('g','r')
    features = []
    for i,col in enumerate(color):
        histr = cv2.calcHist([image],[i+1],mask,[256],[20,250])
        histr = histr/np.sum(histr)
        f = [np.mean(histr), np.std(histr)]

        # TODO : need right feature 
        features.append(np.argmax(histr))
        features.append(np.std(histr))
    print(features)
    return features #[G.mean, G.mean, R.mean, R.std]

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
    # d = extract_rgb_features(image, b)
    d = extract_histogram_features(image, b)
    # print(d)
    p = NN.predict(d)
    plot_histogram([image],[b])
    show_result(_img, p, d)


def train_NN():
    data = []
    images = []
    banana_contours = []

    for i in range(1,8):
        im = cv2.imread('training_banana/b{}.png'.format(i))
        b, _img = extract_banana(im)
        # d = extract_rgb_features(im, b)
        d = extract_histogram_features(im, b)
        d.append(i)
        data.append(d)

        # for plotting
        images.append(im)
        banana_contours.append(b)

    plot_histogram(images, banana_contours)

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

def plot_histogram(images, bananas):
    fig = plt.figure()
    color = ('b','g','r')
    for i in range(len(images)):
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        mask = banana_mask(images[i], bananas[i])
        histr = cv2.calcHist([gray],[0],mask,[256],[20,250])
        plt.subplot(len(images), len(color) + 1 , (1+i)+(len(color)*i)+ len(color))
        print(np.mean(histr),np.std(histr))

        plt.plot(histr,color = 'gray')
        plt.xlim([0,256])

        for j,col in enumerate(color):
            histr = cv2.calcHist([images[i]],[j],mask,[256],[20,250])
            plt.subplot(len(images), len(color) + 1,(1+i)+(len(color)*i) + j)
            plt.plot(histr,color = col)
            plt.xlim([0,256])
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
    # main()