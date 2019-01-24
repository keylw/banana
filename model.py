import numpy as np
import csv
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class NN():
    X = []
    Y = []
    @staticmethod
    def predict(banana):
        label = -1
        nn_distance = sys.maxsize
        for i in range(len(NN.X)):
            dist = NN.rgb_euclidean_distance(banana, NN.X[i])
            if dist < nn_distance : 
                nn_distance = dist
                label = NN.Y[i]
        print(label)
        return label

    @staticmethod
    def rgb_euclidean_distance(x,y):
        R_m = (x[0] - y[0])**2
        G_m = (x[1] - y[1])**2
        B_m = (x[2] - y[2])**2
        return np.sqrt(R_m + G_m + B_m) 

    @staticmethod
    def train(datafile = "training_banana/banana.csv"):
        with open(datafile, 'r') as myfile:
            csv_reader = csv.reader(myfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                d = [row[0],row[1],row[2],row[3],row[3]]
                NN.X.append(d)
                NN.Y.append(r4w[4])


class Bayer():
    model = GaussianNB()

    @staticmethod
    def predict(banana_feature):
        # print(banana_feature)
        return Bayer.model.predict([banana_feature])

    @staticmethod
    def train(datafile = "training_banana/banana.csv"):
        X = []
        Y = []

        with open(datafile, 'r') as myfile:
            csv_reader = csv.reader(myfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                d = [row[0],row[1],row[2],row[3]]
                X.append(d)
                Y.append(row[4])
        Bayer.model.fit(X,Y)

class MLP():
    model = MLPClassifier()

    @staticmethod
    def predict(banana_feature):
        # print(banana_feature)
        return Bayer.model.predict([banana_feature])

    @staticmethod
    def train(datafile = "training_banana/banana.csv"):
        X = []
        Y = []

        with open(datafile, 'r') as myfile:
            csv_reader = csv.reader(myfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                d = [row[0],row[1],row[2],row[3]]
                X.append(d)
                Y.append(row[4])
        Bayer.model.fit(X,Y)