import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import glob, sys, pickle
from time import time

def create_baseline():
    # create model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5,5), activation='relu',input_shape=(100,100,1)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(265, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    seed = int(time())
    np.random.seed(seed)

    datapath = sys.argv[1]

    files = glob.glob(datapath+"**/highway_data.npy")
    X = []
    Y = []
    for file in files:
        data = np.load(file)
        for point in data:
            X.append(list(point[0]))
            Y.append(point[1])
    X = np.array(X)
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    Y = np.array(Y)
    shuffler = np.arange(len(X))
    np.random.shuffle(shuffler)
    X = X[shuffler]
    Y = Y[shuffler]
    model = create_baseline()
    model.fit(X,Y,batch_size=32,epochs=100,verbose=1,shuffle=True,validation_split=.1)
    model.save('highway_model.h5')
