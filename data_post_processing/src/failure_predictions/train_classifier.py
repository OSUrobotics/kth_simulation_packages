import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.misc import imread
from keras.optimizers import Adam
import glob,sys,pickle, os
from time import time, strftime


def create_baseline(window_size):
    # create model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5,5), activation='relu',input_shape=(window_size,window_size,1)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=.01), metrics=['accuracy'])
    return model

if __name__ == "__main__":
    seed = int(time())
    np.random.seed(seed)

    datapath = sys.argv[1]
    model_folder = sys.argv[2]
    # loading the dataset
    files = glob.glob(datapath+'**/failure_training_data_5.npz')
    X = []
    Y = []
    np.random.shuffle(files)
    training_files = files[:int(len(files)*.9)]
    validation_files = files[int(len(files)*.9):]
    for file in training_files:
        data = np.load(file)
        for point in data:
            X.append((point[0]/255.0))
            Y.append(int(point[1]))
    X = np.array(X)
    # print(X.mean())
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    Y = np.array(Y)
    shuffler = np.arange(len(X))
    np.random.shuffle(shuffler)
    X = X[shuffler]
    Y = Y[shuffler]

    model = create_baseline(window_size=90)
    model.fit(X,Y,batch_size=32,epochs=100,verbose=1,shuffle=True)
    date = strftime("%d_%m_%Y_%H_%M_%S")
    if not os.path.exists(model_folder + date):
        os.makedirs(model_folder + date)
    model.save(model_folder + date+'/failure_model.h5')
    with open(model_folder + date+'/training_validation_split.txt', 'wb') as f:
        f.write('training files\n')
        for file in training_files:
            f.write(file + '\n')
        f.write('validation files\n')
        for file in validation_files:
            f.write(file + '\n')
