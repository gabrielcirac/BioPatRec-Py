import NetLib
import FeatureExtractor
import numpy



class Stats:
    def __init__(self, num_rep):
        self.num_rep = num_rep

    def resulter(self):
        features = 136
        acc_mean = 0
        acc_final = 0
        tempo_mean = 0
        tempo_final = 0
        std_mean = []
        std_time = []



        for j in range(0, self.num_rep):

            for i in range(1, 18):
                print("Pearson: ", i)
                print("Repetition: ", j)
                data = FeatureExtractor.DataExtractor_CNN2D(num_peaple=i, features=features)
                x_train, y_train, x_test, y_test = data.generator()
                tempo, acc = NetLib.Net3(x_train, y_train, x_test, y_test, features).train()
                acc_mean = acc_mean + acc
                tempo_mean = tempo_mean + tempo


            acc_mean = acc_mean/17
            acc_final = acc_final + acc_mean

            tempo_mean = tempo_mean/17
            tempo_final = tempo_final + tempo_mean

            std_mean.append(acc_mean)
            std_time.append(tempo_mean)
            acc_mean = 0
            tempo_mean = 0

        print("Acuracy: ", acc_final/self.num_rep, "STD: ", numpy.array(std_mean).std())
        print("Tempo: ", tempo_final/self.num_rep, "STD: ", numpy.array(std_time).std())

        return 0


obj = Stats(10)
obj.resulter()