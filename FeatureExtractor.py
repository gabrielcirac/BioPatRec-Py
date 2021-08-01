import pandas as pd
import talib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
os.chdir("./Treino/")

class DataExtractor_LSTM:
    def __init__(self, num_peaple, features):
        self.num_peaple = num_peaple
        self.features = features

    def generator(self):
        data = pd.read_csv('Treino' + str(self.num_peaple) + '.csv', dtype={'val': str}, header=None)
        pd.options.mode.chained_assignment = None
        x_train = pd.DataFrame(data)


        # x_train = x_train.drop(x_train.columns[0], axis=1)
        y_train = x_train.iloc[:, 136]
        x_train = x_train.drop(x_train.columns[136], axis=1)
        x_train = x_train.iloc[:, 0:self.features]

        for i in range(0, self.features):
            x_train.iloc[:, i] = (talib.KAMA(x_train.iloc[:, i], timeperiod=30))

        data = pd.read_csv('Teste' + str(self.num_peaple) + '.csv', dtype={'val': str}, header=None)
        pd.options.mode.chained_assignment = None
        x_test = pd.DataFrame(data)

        # x_test = x_test.drop(x_test.columns[0], axis=1)
        y_test = x_test.iloc[:, 136]
        x_test = x_test.drop(x_test.columns[136], axis=1)
        x_test = x_test.iloc[:, 0:self.features]

        # # plt.plot(x_test.iloc[:, 7], label='Feature')
        for i in range(0, self.features):
            x_test.iloc[:, i] = (talib.KAMA(x_test.iloc[:, i], timeperiod=30))

        # plt.plot(x_test.iloc[:, 7], label='Filtered')
        # plt.legend()
        # plt.show()

        x_test = x_test[30:]
        y_test = y_test[30:]

        x_train = x_train[30:]
        y_test = y_test[30:]

        frames = [x_train, x_test]
        x_total = pd.concat(frames)
        x_total = x_total.reset_index(drop=True)

        frames = [y_train, y_test]
        y_total = pd.concat(frames)
        y_total = y_total.reset_index(drop=True)

        # x_total = x_total[100:]
        # y_total = y_total[100:]

        # import seaborn as sns
        # sns.set(color_codes=True)
        # sns.distplot(x_test.iloc[:, 100], label='Feature 100')
        # plt.title('Distribution before Normalization')
        # plt.show()

        scale = preprocessing.QuantileTransformer(n_quantiles=100)
        x_total.iloc[:, 0:self.features] = scale.fit_transform(x_total.iloc[:, 0:self.features])


        x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.4, random_state=42, shuffle=True)

        # import seaborn as sns
        # sns.set(color_codes=True)
        # sns.distplot(x_test.iloc[:, 100], label='Feature 100');
        # plt.title('Distribution after Normalization')
        # plt.show()

        # reshape input to be 3D [samples, timesteps, features]
        x_train = x_train.values
        x_train = x_train.reshape((x_train.shape[0], 1, int(x_train.shape[1]/1)))

        x_test = x_test.values
        x_test = x_test.reshape((x_test.shape[0], 1, int(x_test.shape[1]/1)))


        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_Y = encoder.transform(y_train)
        # convert integers to dummy variables (i.e. one hot encoded)
        y_train = to_categorical(encoded_Y)

        encoder = LabelEncoder()
        encoder.fit(y_test)
        encoded_Y = encoder.transform(y_test)
        # convert integers to dummy variables (i.e. one hot encoded)
        y_test = to_categorical(encoded_Y)



        return x_train, y_train, x_test, y_test


class DataExtractor_CNN1D:
    def __init__(self, num_peaple, features):
        self.num_peaple = num_peaple
        self.features = features

    def generator(self):
        data = pd.read_csv('Treino' + str(self.num_peaple) + '.csv', dtype={'val': str}, header=None)
        pd.options.mode.chained_assignment = None
        x_train = pd.DataFrame(data)


        # x_train = x_train.drop(x_train.columns[0], axis=1)
        y_train = x_train.iloc[:, 136]
        x_train = x_train.drop(x_train.columns[136], axis=1)
        x_train = x_train.iloc[:, 0:self.features]

        for i in range(0, self.features):
            x_train.iloc[:, i] = (talib.KAMA(x_train.iloc[:, i], timeperiod=30))

        data = pd.read_csv('Teste' + str(self.num_peaple) + '.csv', dtype={'val': str}, header=None)
        pd.options.mode.chained_assignment = None
        x_test = pd.DataFrame(data)

        # x_test = x_test.drop(x_test.columns[0], axis=1)
        y_test = x_test.iloc[:, 136]
        x_test = x_test.drop(x_test.columns[136], axis=1)
        x_test = x_test.iloc[:, 0:self.features]

        # # plt.plot(x_test.iloc[:, i])
        for i in range(0, self.features):
            x_test.iloc[:, i] = (talib.KAMA(x_test.iloc[:, i], timeperiod=30))

        # plt.plot(x_test.iloc[:, i])
        # plt.show()

        x_test = x_test[30:]
        y_test = y_test[30:]

        x_train = x_train[30:]
        y_test = y_test[30:]

        frames = [x_train, x_test]
        x_total = pd.concat(frames)
        x_total = x_total.reset_index(drop=True)

        frames = [y_train, y_test]
        y_total = pd.concat(frames)
        y_total = y_total.reset_index(drop=True)

        # x_total = x_total[100:]
        # y_total = y_total[100:]

        scale = preprocessing.QuantileTransformer(n_quantiles=100)
        x_total.iloc[:, 0:self.features] = scale.fit_transform(x_total.iloc[:, 0:self.features])

        x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.4, random_state=13, shuffle=True)

        # reshape input to be 3D [samples, timesteps, features]
        x_train = x_train.values
        x_train = x_train.reshape((x_train.shape[0], 1, int(x_train.shape[1]/1)))

        x_test = x_test.values
        x_test = x_test.reshape((x_test.shape[0], 1, int(x_test.shape[1]/1)))


        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_Y = encoder.transform(y_train)
        # convert integers to dummy variables (i.e. one hot encoded)
        y_train = to_categorical(encoded_Y)

        encoder = LabelEncoder()
        encoder.fit(y_test)
        encoded_Y = encoder.transform(y_test)
        # convert integers to dummy variables (i.e. one hot encoded)
        y_test = to_categorical(encoded_Y)

        y_train = y_train.reshape((len(y_train), 1, 27))
        y_test = y_test.reshape((len(y_test), 1, 27))

        return x_train, y_train, x_test, y_test

class DataExtractor_CNN2D:
    def __init__(self, num_peaple, features):
        self.num_peaple = num_peaple
        self.features = features

    def generator(self):
        data = pd.read_csv('Treino' + str(self.num_peaple) + '.csv', dtype={'val': str}, header=None)
        pd.options.mode.chained_assignment = None
        x_train = pd.DataFrame(data)


        # x_train = x_train.drop(x_train.columns[0], axis=1)
        y_train = x_train.iloc[:, 136]
        x_train = x_train.drop(x_train.columns[136], axis=1)
        x_train = x_train.iloc[:, 0:self.features]

        for i in range(0, self.features):
            x_train.iloc[:, i] = (talib.KAMA(x_train.iloc[:, i], timeperiod=30))

        data = pd.read_csv('Teste' + str(self.num_peaple) + '.csv', dtype={'val': str}, header=None)
        pd.options.mode.chained_assignment = None
        x_test = pd.DataFrame(data)

        # x_test = x_test.drop(x_test.columns[0], axis=1)
        y_test = x_test.iloc[:, 136]
        x_test = x_test.drop(x_test.columns[136], axis=1)
        x_test = x_test.iloc[:, 0:self.features]

        # # plt.plot(x_test.iloc[:, i])
        for i in range(0, self.features):
            x_test.iloc[:, i] = (talib.KAMA(x_test.iloc[:, i], timeperiod=30))

        # plt.plot(x_test.iloc[:, i])
        # plt.show()

        x_test = x_test[30:]
        y_test = y_test[30:]

        x_train = x_train[30:]
        y_test = y_test[30:]

        frames = [x_train, x_test]

        x_total = pd.concat(frames)
        x_total = x_total.reset_index(drop=True)

        frames = [y_train, y_test]
        y_total = pd.concat(frames)
        y_total = y_total.reset_index(drop=True)

        # x_total = x_total[100:]
        # y_total = y_total[100:]

        scale = preprocessing.QuantileTransformer(n_quantiles=100)
        x_total.iloc[:, 0:self.features] = scale.fit_transform(x_total.iloc[:, 0:self.features])

        x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.4, random_state=13, shuffle=True)

        # reshape input to be 3D [samples, timesteps, features]
        x_train = x_train.values
        x_train = x_train.reshape((x_train.shape[0], 8, int(x_train.shape[1]/8), 1))

        x_test = x_test.values
        x_test = x_test.reshape((x_test.shape[0], 8, int(x_test.shape[1]/8), 1))


        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_Y = encoder.transform(y_train)
        # convert integers to dummy variables (i.e. one hot encoded)
        y_train = to_categorical(encoded_Y)

        encoder = LabelEncoder()
        encoder.fit(y_test)
        encoded_Y = encoder.transform(y_test)
        # convert integers to dummy variables (i.e. one hot encoded)
        y_test = to_categorical(encoded_Y)

        # y_train = y_train.reshape((len(y_train), 1, 27))
        # y_test = y_test.reshape((len(y_test), 1, 27))

        return x_train, y_train, x_test, y_test

class DataExtractor_XB:
    def __init__(self, num_peaple, features):
        self.num_peaple = num_peaple
        self.features = features

    def generator(self):
        data = pd.read_csv('Treino' + str(self.num_peaple) + '.csv', dtype={'val': str}, header=None)
        pd.options.mode.chained_assignment = None
        x_train = pd.DataFrame(data)


        # x_train = x_train.drop(x_train.columns[0], axis=1)
        y_train = x_train.iloc[:, 136]
        x_train = x_train.drop(x_train.columns[136], axis=1)
        x_train = x_train.iloc[:, 0:self.features]

        for i in range(0, self.features):
            x_train.iloc[:, i] = (talib.KAMA(x_train.iloc[:, i], timeperiod=30))

        data = pd.read_csv('Teste' + str(self.num_peaple) + '.csv', dtype={'val': str}, header=None)
        pd.options.mode.chained_assignment = None
        x_test = pd.DataFrame(data)

        # x_test = x_test.drop(x_test.columns[0], axis=1)
        y_test = x_test.iloc[:, 136]
        x_test = x_test.drop(x_test.columns[136], axis=1)
        x_test = x_test.iloc[:, 0:self.features]

        # plt.plot(x_test.iloc[:, i])
        for i in range(0, self.features):
            x_test.iloc[:, i] = (talib.KAMA(x_test.iloc[:, i], timeperiod=30))

        # plt.plot(x_test.iloc[:, i])
        # plt.show()

        x_test = x_test[94:]
        y_test = y_test[94:]

        x_train = x_train[73:]
        y_test = y_test[73:]

        frames = [x_train, x_test]

        x_total = pd.concat(frames)
        x_total = x_total.reset_index(drop=True)

        frames = [y_train, y_test]
        y_total = pd.concat(frames)
        y_total = y_total.reset_index(drop=True)

        x_total = x_total[100:]
        y_total = y_total[100:]

        scale = preprocessing.QuantileTransformer(n_quantiles=100)
        x_total.iloc[:, 0:self.features] = scale.fit_transform(x_total.iloc[:, 0:self.features])

        x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.4, random_state=13, shuffle=True)

        return x_train, y_train, x_test, y_test