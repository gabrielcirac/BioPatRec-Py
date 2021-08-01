from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Conv2D, Flatten, MaxPooling1D, GRU
import tensorflow as tf
import time
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


class Net1:
    def __init__(self, x_train, y_train, x_test, y_test, num_features):
        self.X_test = x_test
        self.Y_test = y_test
        self.X_Train = x_train
        self.Y_Train = y_train
        self.num_Features = num_features


    def train(self):

        # Configure keras to use GPU
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                        device_count={'GPU': 1, 'CPU': 12})
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        #Modelo Criativo
        model = Sequential()
        # tf.compat.v1.keras.layers.CuDNNLSTM
        model.add(LSTM(100, recurrent_activation='sigmoid', input_shape=(self.X_Train.shape[1], self.X_Train.shape[2]), unroll=False))
        model.add(Dense(27, activation='softmax'))
        opt = tf.optimizers.Adam(learning_rate=0.05,
               beta_1=0.99,
               beta_2=0.999,
               epsilon=1e-3,)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

        epochs = 200
        batch_size = int(len(self.Y_Train) / 4)

        start_time = time.time()
        model.fit(self.X_Train, self.Y_Train, epochs=epochs, batch_size=batch_size, validation_split=0, workers=-1,
                  use_multiprocessing=True, verbose=0, )
        tempo = (time.time() - start_time)
        print("--- %s seconds ---" % tempo)

        scores = model.evaluate(self.X_test, self.Y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        return tempo, (scores[1] * 100)


class Net2:
    def __init__(self, x_train, y_train, x_test, y_test, num_features):
        self.X_test = x_test
        self.Y_test = y_test
        self.X_Train = x_train
        self.Y_Train = y_train
        self.num_Features = num_features


    def train(self):

        # Configure keras to use GPU
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                        device_count={'GPU': 1, 'CPU': 12})
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        print('Aqui')
        print(self.X_Train.shape, self.Y_Train.shape, self.X_test.shape, self.Y_test.shape)

        #Modelo Criativo
        model = Sequential()
        # tf.compat.v1.keras.layers.CuDNNLSTM
        model.add(Conv1D(filters=136, kernel_size=1, activation='tanh', input_shape=(self.X_Train.shape[1], self.X_Train.shape[2])))

        model.add(Dense(27, activation='softmax'))
        opt = tf.optimizers.Adam(learning_rate=0.05,
               beta_1=0.99,
               beta_2=0.999,
               epsilon=1e-3,)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])


        epochs = 100
        batch_size = int(len(self.Y_Train) / 4)

        start_time = time.time()
        model.fit(self.X_Train, self.Y_Train, epochs=epochs, batch_size=batch_size, validation_split=0,
                  use_multiprocessing=True, verbose=0, workers=-1 )
        tempo = (time.time() - start_time)
        print("--- %s seconds ---" % tempo)

        scores = model.evaluate(self.X_test, self.Y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        return tempo, (scores[1] * 100)


class Net3:
    def __init__(self, x_train, y_train, x_test, y_test, num_features):
        self.X_test = x_test
        self.Y_test = y_test
        self.X_Train = x_train
        self.Y_Train = y_train
        self.num_Features = num_features


    def train(self):

        # Configure keras to use GPU
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                        device_count={'GPU': 1, 'CPU': 12})
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)


        # print(self.X_Train.shape, self.Y_Train.shape, self.X_test.shape, self.Y_test.shape)

        #Modelo Criativo
        model = Sequential()
        # tf.compat.v1.keras.layers.CuDNNLSTM
        model.add(Conv2D(filters=34, kernel_size=8, activation='softmax', input_shape=(8,17,1)))

        model.add(Flatten())
        model.add(Dense(self.Y_test.shape[1], activation='softmax'))
        opt = tf.optimizers.Adam(learning_rate=0.05,
               beta_1=0.99,
               beta_2=0.999,
               epsilon=1e-3,)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # print(model.summary())

        epochs = 100
        batch_size = int(len(self.Y_Train) / 4)

        start_time = time.time()
        model.fit(self.X_Train, self.Y_Train, epochs=epochs, batch_size=batch_size, validation_split=0,
                  use_multiprocessing=False, verbose=0, )
        tempo = (time.time() - start_time)
        print("--- %s seconds ---" % tempo)

        scores = model.evaluate(self.X_test, self.Y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        import matplotlib.pyplot as plt
        for layer in model.layers:
            if 'conv' in layer.name:
                weights, bias = layer.get_weights()

                # normalize filter values between  0 and 1 for visualization
                f_min, f_max = weights.min(), weights.max()
                filters = (weights - f_min) / (f_max - f_min)
                print(filters.shape[3])
                filter_cnt = 1

                # # plotting all the filters
                # for i in range(filters.shape[3]):
                #     # get the filters
                #     filt = filters[:, :, :, i]
                #     # plotting each of the channel, color image RGB channels
                #     for j in range(filters.shape[0]):
                #         ax = plt.subplot(filters.shape[3], filters.shape[0], filter_cnt)
                #         ax.set_xticks([])
                #         ax.set_yticks([])
                #         plt.imshow(filt[:, :, 0])
                #         filter_cnt += 1
                # plt.show()

        return tempo, (scores[1] * 100)

class Net4:
    def __init__(self, x_train, y_train, x_test, y_test, num_features):
        self.X_test = x_test
        self.Y_test = y_test
        self.X_Train = x_train
        self.Y_Train = y_train
        self.num_Features = num_features


    def train(self):
        clf = AdaBoostClassifier(ExtraTreeClassifier(max_depth=15),
                                 n_estimators=300, learning_rate=0.01)

        start_time = time.time()
        clf.fit(self.X_Train, self.Y_Train)
        tempo = time.time() - start_time
        Pred = clf.predict(self.X_test)
        score = (accuracy_score(self.Y_test, Pred))

        print(score, tempo)
        return tempo, score


class Net5:
    def __init__(self, x_train, y_train, x_test, y_test, num_features):
        self.X_test = x_test
        self.Y_test = y_test
        self.X_Train = x_train
        self.Y_Train = y_train
        self.num_Features = num_features


    def train(self):

        # Configure keras to use GPU
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                        device_count={'GPU': 1, 'CPU': 12})
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        #Modelo Criativo
        model = Sequential()
        # tf.compat.v1.keras.layers.CuDNNLSTM
        # model.add(LSTM(100, recurrent_activation='sigmoid', input_shape=(self.X_Train.shape[1], self.X_Train.shape[2]), unroll=False))
        model.add(GRU(50, recurrent_activation='sigmoid'))
        model.add(Dense(27, activation='softmax'))
        opt = tf.optimizers.Adam(learning_rate=0.05,
               beta_1=0.99,
               beta_2=0.999,
               epsilon=1e-3,)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

        epochs = 200
        batch_size = int(len(self.Y_Train) / 4)

        start_time = time.time()
        model.fit(self.X_Train, self.Y_Train, epochs=epochs, batch_size=batch_size, validation_split=0, workers=-1,
                  use_multiprocessing=True, verbose=0, )
        tempo = (time.time() - start_time)
        print("--- %s seconds ---" % tempo)

        scores = model.evaluate(self.X_test, self.Y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        return tempo, (scores[1] * 100)