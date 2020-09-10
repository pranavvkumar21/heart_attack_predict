import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

def model1(input_shape, layers):
    X_in = Input((input_shape,))
    X = X_in
    for i in range(input_shape,0,-(input_shape//layers)):
        X = Dense(i)(X)
        Y = Dense(1, activation='sigmoid')(X)
    model = Model(inputs = X_in, outputs = Y, name='basic1')
    #model.summary()
    return model
