import glob
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib.pyplot as plt
import sys


def performKNN(PATH, n=10, end_k=30):
    prediction_count_for_k = {}
    mean_accuracy_for_k = {}
    test_index = 1
    # errorRates stores error rates for n-fold cross validation
    errorRates = {}
    while test_index <= n:
        #     Store test data in a dataframe
        data_file = pd.read_csv(
            PATH + "data" + str(test_index) + ".csv", header=None)
        label_file = pd.read_csv(
            PATH + "labels" + str(test_index) + ".csv", header=None)
        test_data = pd.concat([data_file, label_file],
                              axis=1, ignore_index=True)
    #    Store the training data in a dataframe
        training_data = pd.DataFrame()
        for current_index in range(1, n + 1):
            if current_index != test_index:
                data_file = pd.read_csv(
                    PATH + "data" + str(current_index) + ".csv", header=None)
                label_file = pd.read_csv(
                    PATH + "labels" + str(current_index) + ".csv", header=None)
                df = pd.concat([data_file, label_file],
                               axis=1, ignore_index=True)
                training_data = training_data.append(df, ignore_index=True)
    #   KNN Logic
        actualLabel = []
        predictedLabel = []
        for x in range(len(test_data)):
            neighbors = getNeighbors(training_data, test_data.iloc[x], end_k)
            findPrediction(
                neighbors, test_data.iloc[x, -1], prediction_count_for_k)
        test_index += 1

    for key, value in prediction_count_for_k.items():
        prediction_count_for_k[key] = (sum(
            prediction_count_for_k[key]) / len(prediction_count_for_k[key])) * 100

    print(
        "Plot shows the mean accuracy for k=[1..30], across a 10 fold cross validation")
    plt.plot(prediction_count_for_k.keys(),
             prediction_count_for_k.values(), marker='o')
    plt.title("Mean Accuracy (%) Vs. K")
    plt.xlabel("Value of K")
    plt.ylabel("Mean accuracy (%)")
    plt.show()


def getNeighbors(training_data, test_instance, end_k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_data)):
        dist = findDistance(test_instance, training_data.iloc[x], length)
        distances.append((training_data.iloc[x], dist))
    distances = sorted(distances, key=lambda tup: tup[1])
    neighbors = []
    for x in range(end_k):
        neighbors.append(distances[x][0].iloc[-1])
    return neighbors


def findDistance(test_instance, train_instance, length):
    distance = 0
    for x in range(length):
        distance += pow((test_instance[x] - train_instance[x]), 2)
    return math.sqrt(distance)


def findPrediction(neighbors, test_data_class, prediction_count_for_k):
    # similar logic to the function above
    labelCount = {}
    for x in range(len(neighbors)):
        class_name = neighbors[x]
        if class_name in labelCount:
            labelCount[class_name] += 1
        else:
            labelCount[class_name] = 1
        sortedLabels = sorted(labelCount.items(),
                              key=lambda kv: kv[1], reverse=True)
        predicted_class = sortedLabels[0][0]
        # for a specific case of k, append 1 successful prediction, else append 0
        if predicted_class == test_data_class:
            prediction_count_for_k.setdefault(x + 1, []).append(1)
        else:
            prediction_count_for_k.setdefault(x + 1, []).append(0)


if __name__ == '__main__':
    PATH = str(sys.argv[1])
    performKNN(PATH, 10, 30)
