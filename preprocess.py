#! /usr/bin/env python3
import csv
import numpy as np
from sklearn.utils import shuffle


def getdata(filename):
    File = open(filename)
    data = np.array(list(csv.reader(File)))
    return data

def getxandy(*args):
    X  = np.concatenate((args), axis=0)
    Y = []
    for idx,arg in enumerate(args):
        y = np.ones((arg.shape[0],1))*idx
        Y.append(y)
    Y = np.concatenate((Y), axis = 0)
    return X,Y

def savefile(**kwargs):
    for key,File in kwargs.items():
        np.save(key+".npy",File)



if __name__ == "__main__":

    normal = getdata("ptbdb_normal.csv")
    abnormal = getdata("ptbdb_abnormal.csv")
    X,Y = getxandy(normal,abnormal)
    X,Y = shuffle(X,Y)
    savefile(Xtrain = X, Ytrain = Y)
