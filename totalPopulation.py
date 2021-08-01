# coding: utf-8

__author__ = 'gabrielcirac'

import time
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM, GRU, Conv2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import talib
import os
import glob
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
os.chdir("./Treino/")


class genericLSTM:

    def __init__(self, Name_csv):
        self.Name_csv = Name_csv

    def train_net(self):
        # Configure keras to use GPU
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                          device_count={'GPU': 1, 'CPU': 8})
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        dataset = pd.read_csv(self.Name_csv, dtype={'val': str}, index_col=False)
        df = pd.DataFrame(dataset, columns=None)

        df = df.dropna()

        # Check distribuition before normalization
        sns.set(color_codes=True)
        sns.distplot(df.iloc[:, 100], label='Feature 100')
        # plt.title('Feature Distribution before Normalization - Total population')
        plt.show()

        data = df.values

        X = data[:, 0:136]
        Y = data[:, 136]
        Y = pd.to_numeric(Y, downcast='integer')

        for i in range(0, 136):
            X[:, i] = (talib.KAMA(X[:, i], timeperiod=30))

        # First Remove NaNs
        X = X[30:]
        Y = Y[30:]

        scale = preprocessing.QuantileTransformer(n_quantiles=100)
        X = scale.fit_transform(X)

        embedding = PCA(n_components=130, svd_solver='randomized',
                        whiten=True)
        X = embedding.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=4, shuffle=True)

        # Check distribuition after normalization
        sns.set(color_codes=True)
        sns.distplot(X_test[:, 100], label='Feature 100');
        # plt.title('Distribution after Normalization')
        plt.show()

        # reshape input to be 3D [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], 1, int(X_train.shape[1] / 1)))
        X_test = X_test.reshape((X_test.shape[0], 1, int(X_test.shape[1] / 1)))

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        print('Classes: ', y_test.shape[1])
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        print((X_train.shape[1], X_train.shape[2]))

        # Modelo Criativo
        model = Sequential()
        # tf.compat.v1.keras.layers.CuDNNLSTM
        model.add(LSTM(units=100, unroll=True, return_sequences=True, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
        # model.add(Dropout(0.2))
        # model.add(Conv1D(filters=25, kernel_size=1, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
        # model.add(Dropout(0.2))
        # model.add(Conv1D(filters=15, kernel_size=1, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Flatten())
        # model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(y_test.shape[1], activation='softmax'))
        opt = tf.optimizers.RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

        epochs = 20
        batch_size = int(len(y_train) / 128)

        start_time = time.time()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                  use_multiprocessing=True, verbose=1, )
        tempo = (time.time() - start_time)
        print("--- %s seconds ---" % tempo)
        start_time = time.time()
        print(len(X_test))
        scores = model.evaluate(X_test, y_test, verbose=1)
        tempo = (time.time() - start_time)
        print("--- %s seconds ---" % tempo)
        print("Accuracy: %.2f%%" % (scores[1] * 100))




class noisyLSTM:

    def __init__(self, Name_csv):
        self.Name_csv = Name_csv

    def train_net(self):
        # Configure keras to use GPU
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                          device_count={'GPU': 1, 'CPU': 12})
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        dataset = pd.read_csv(self.Name_csv, dtype={'val': str}, index_col=False)
        df = pd.DataFrame(dataset, columns=None)
        df = df.dropna()

        # Check distribuition before normalization
        # sns.set(color_codes=True)
        # sns.distplot(df.iloc[:, 100], label='Feature 100')
        # # plt.title('Feature Distribution before Normalization - Total population')
        # plt.show()


        # df = df.iloc[::2]
        # print(df[df.duplicated(['3'], keep=False)])
        data = df.values

        X = data[:, 0:136]
        Y = data[:, 136]
        Y = pd.to_numeric(Y, downcast='integer')

        for i in range(0, 136):
            X[:, i] = (talib.KAMA(X[:, i], timeperiod=30))

        # First Remove NaNs
        X = X[30:]
        Y = Y[30:]

        scale = preprocessing.QuantileTransformer(n_quantiles=100)
        X = scale.fit_transform(X)

        embedding = PCA(n_components=130, svd_solver='randomized',
                        whiten=True)
        X = embedding.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=4, shuffle=True)


        # # Check distribuition after normalization
        # sns.set(color_codes=True)
        # sns.distplot(X_test[:, 100], label='Feature 100');
        # # plt.title('Distribution after Normalization')
        # plt.show()


        featureClean = X_test[:, 95]

        import numpy as np
        mu, sigma = 0, 0.5
        # creating a noise with the same dimension as the dataset (2,2)
        noise = np.random.normal(mu, sigma, [X_test.shape[0], X_test.shape[1]])
        X_test = X_test + noise
        noise = X_test[:, 95]



        # plt.plot(X_test[:, 100], label="Noisy Sample")
        # plt.plot(featureClean, label="Clean Sample")
        # plt.legend()
        # plt.title("Difference between a standard feature and a noisy feature - Sigma 0.1")
        # plt.show()
        #
        # # Check distribuition after normalization
        # sns.set(color_codes=True)
        # sns.distplot(X_test[:, 100], label='Feature 100');
        # # plt.title('Distribution after Normalization')
        # plt.show()


        # reshape input to be 3D [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], 1, int(X_train.shape[1] / 1)))
        X_test = X_test.reshape((X_test.shape[0], 1, int(X_test.shape[1] / 1)))

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        print('Classes: ', y_test.shape[1])
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        print((X_train.shape[1], X_train.shape[2]))

        # Modelo Criativo
        model = Sequential()
        # tf.compat.v1.keras.layers.CuDNNLSTM
        model.add(LSTM(units=350, unroll=True, return_sequences=True, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
        # # model.add(Conv1D(filters=25, kernel_size=1, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
        # # model.add(Dropout(0.2))
        # model.add(Flatten())
        # model.add(Conv1D(filters=15, kernel_size=1, activation='tanh'))
        model.add(Flatten())
        # model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(y_test.shape[1], activation='softmax'))
        opt = tf.optimizers.RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

        epochs = 100
        batch_size = int(len(y_train) / 4)

        start_time = time.time()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.30,
                  use_multiprocessing=True, verbose=1, )
        tempo = (time.time() - start_time)
        print("--- %s seconds ---" % tempo)
        start_time = time.time()
        print(len(X_test))
        scores = model.evaluate(X_test, y_test, verbose=1)
        tempo = (time.time() - start_time)
        print("--- %s seconds ---" % tempo)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        fig = plt.figure(figsize=[25, 10])
        ax = fig.add_subplot(111)


        plt.plot(noise[0:200], label="Noisy Sample", color="red")
        plt.plot(featureClean[0:200], label="Clean Sample", color="green")
        plt.title("Difference between a standard feature and a noisy feature - Sigma 0.5")

        ax.text(0, 3.5, 'Accuracy', style='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10}, fontsize=20)

        ax.text(25, 3.5, round((scores[1] * 100),2), style='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10}, fontsize=20)

        red_patch = mpatches.Patch(color='green', label='Normal')
        green_patch = mpatches.Patch(color='red', label='Noisy')
        plt.legend(handles=[red_patch, green_patch])
        plt.show()



class noisyADA:

    def __init__(self, Name_csv):
        self.Name_csv = Name_csv

    def train_net(self):

        dataset = pd.read_csv(self.Name_csv, dtype={'val': str}, index_col=False)
        df = pd.DataFrame(dataset, columns=None)
        df = df.dropna()
        data = df.values

        X = data[:, 0:136]
        Y = data[:, 136]
        Y = pd.to_numeric(Y, downcast='integer')

        for i in range(0, 136):
            X[:, i] = (talib.KAMA(X[:, i], timeperiod=30))

        # First Remove NaNs
        X = X[30:]
        Y = Y[30:]

        scale = preprocessing.QuantileTransformer(n_quantiles=100)
        X = scale.fit_transform(X)

        embedding = PCA(n_components=130, svd_solver='randomized',
                        whiten=True)
        X = embedding.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=3, shuffle=True)


        #Choose 1-130
        featureClean = X_test[:, 95]


        mu, sigma = 0, 0.5
        # creating a noise with the same dimension as the dataset (2,2)
        noise = np.random.normal(mu, sigma, [X_test.shape[0], X_test.shape[1]])
        X_test = X_test + noise
        noise = X_test[:, 95]


        clf = AdaBoostClassifier(ExtraTreesClassifier(max_depth=30, verbose=1, n_jobs=-1),
                                 n_estimators=200, learning_rate=0.05)

        start_time = time.time()
        clf.fit(X_train, y_train)
        time_t = time.time() - start_time
        Pred = clf.predict(X_test)
        score = (accuracy_score(y_test, Pred))
        print("Score:", score)
        print("Time:", time_t)


        fig = plt.figure(figsize=[25, 10])
        ax = fig.add_subplot(111)


        plt.plot(noise[0:200], label="Noisy Sample", color="red")
        plt.plot(featureClean[0:200], label="Clean Sample", color="green")
        plt.title("Difference between a standard feature and a noisy feature - Sigma 0.5")

        ax.text(0, 3.5, 'Accuracy', style='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10}, fontsize=20)

        ax.text(25, 3.5, round(score*100, 4), style='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10}, fontsize=20)

        red_patch = mpatches.Patch(color='green', label='Normal')
        green_patch = mpatches.Patch(color='red', label='Noisy')
        plt.legend(handles=[red_patch, green_patch])
        plt.show()

objeto = noisyADA('Total.csv')
objeto.train_net()

objeto = noisyLSTM('Total.csv')
objeto.train_net()