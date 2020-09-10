#! /usr/bin/env python3
import csv
import numpy as np
from sklearn.utils import shuffle
from models import *
import keras
import tensorflow as tf
from numpy import genfromtxt

def getdata(filename):
    File = open(filename)
    data = genfromtxt(filename, delimiter=",")
    return data

def getxandy(*args):
    X  = np.concatenate((args), axis=0)
    Y = []
    for idx,arg in enumerate(args):
        y = np.ones((arg.shape[0],))*idx
        Y.append(y)
    Y = np.concatenate((Y), axis = 0)
    return X,Y

def savefile(**kwargs):
    for key,File in kwargs.items():
        np.save(key+".npy",File)

def train_data(model, X, Y):
    opt = keras.optimizers.Adam(learning_rate=0.000003)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(X,Y, epochs=epoch, batch_size=batchsize)

def predict(X,Y):
    

if __name__ == "__main__":

    input_shape =188
    layers = 20
    batchsize = 100
    epoch = 200

    normal = getdata("ptbdb_normal.csv")
    abnormal = getdata("ptbdb_abnormal.csv")
    X,Y = getxandy(normal,abnormal)
    X,Y = shuffle(X,Y)

    X_train,Y_train = X[0:10000],Y[0:10000]
    X_test, Y_test = X[10000:],Y[10000:]
    #print(type(X[0,1]))
    #savefile(Xtrain = X, Ytrain = Y)
    model = model1(188,20)
    train_data(model, X_train, Y_train)
