# ------------------------------------------------------------------------
# Project: 2017 Seed cup round 1
# Author: Jiyang Qi, Zhihao Wang, Yue Pan
# GitHub: https://github.com/zxc479773533/2017-Seed-Cup-Round-1.git
# Module: Main train and predict functions based on DNN
# ------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def loadData(data_file, win_rate):

    # load match information
    with open(data_file, 'r', encoding='utf-8') as fp:
        fp.readline()
        lines = [line.split(',') for line in fp.readlines()]

    # get data
    data_set = []
    label_set = []

    for line in lines:

        # expand team information
        newdata = []

        # get win and lose
        line[2] = line[2].strip()
        line[3] = line[3].strip()
        index2 = line[2].find('胜')
        index3 = line[3].find('胜')
        newdata += [
            int(line[2][:index2]),
            int(line[2][index2+1:-1]),
            int(line[3][:index3]),
            int(line[3][index3+1:-1])
        ]
        newdata.append((win_rate[int(line[0])][1] - win_rate[int(line[1])][0]) * 30)
        newdata.append((win_rate[int(line[0])][2] - win_rate[int(line[1])][2]) * 30)
        data_set.append(newdata)

    # if there is match result
    if len(lines[0]) == 5:
        for line in lines:
            score_data = line[4].split(':')
            diff = int(score_data[0]) - int(score_data[1])
            if diff > 0:
                # if (0, 4) 0; if [4, 11) 1; if [11, ...) 2
                newlabel = 0 if diff < 4 else (1 if diff < 11 else 2)
            else:
                # if (-4, 0] 3; if (-11, -4] 4; if (..., -11] 5
                newlabel = 3 if diff > -4 else (4 if diff > -11 else 5)
            label_set.append(newlabel)

    return data_set, label_set


def train(training_set, label_set):

    # PCA, the n is adjustable parameters, representing the dimensions
    n = None
    print(n)
    pca = PCA(n_components = n)
    training_set = pca.fit_transform(training_set)
    print(pca.explained_variance_ratio_)

    # specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column(
        "x", shape = [len(training_set[0])])]
    print(feature_columns)
    # build 1 layer DNN with 5 units respectively, the hidden units is adjustable parameters
    hidden_units = [5]
    print(hidden_units)
    classifier = tf.estimator.DNNClassifier(feature_columns = feature_columns,
                                            hidden_units =  hidden_units,
                                            n_classes = 6,
                                            model_dir = "./seedcup_model")
    
    # define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": np.array(training_set[:6000])},
        y = np.array(label_set[:6000]),
        num_epochs = None,
        shuffle = True
    )

    # train model, the number in range and steps is adjustable parameters
    for k in range(1):
        classifier.train(input_fn = train_input_fn, steps = 38000)

        # define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": np.array(training_set[:6000])},
            y = np.array(label_set[:6000]).T,
            num_epochs = 1,
            shuffle = False
        )

        # evaluate accuracy.
        accuracy_score = classifier.evaluate(input_fn = test_input_fn)["accuracy"]

        print("\nTrain Accuracy: {0:f}\n".format(accuracy_score))

        # define the test inputs
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": np.array(training_set[6000:])},
            num_epochs = 1,
            shuffle = False
        )

        # evaluate accuracy.
        predictions = list(classifier.predict(input_fn=predict_input_fn))
        classes = [p['classes'] for p in predictions]

        # define the labels for test
        test_label_set = label_set[6000:]
        length = len(classes)
        right = 0
        for i in range(length):
            if (int(classes[i][0].decode('utf8')) - 2.5) * (test_label_set[i] - 2.5) > 0:
                right += 1

        print('train steps number: ', 25000 + (k + 1) * 1000)
        print("New Samples, Accuracy: {}\n".format(right / length))
    return classifier, pca


def predict(classifier, win_rate, pca):

    # Classify two new flower samples.
    new_samples, labels = loadData('../matchDataTest.csv', win_rate)
    new_samples = pca.transform(new_samples)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": np.array(new_samples)},
        num_epochs = 1,
        shuffle = False
    )

    predictions = list(classifier.predict(input_fn = predict_input_fn))
    probabilities = [p['probabilities'] for p in predictions]
    win_prob = [sum(x[3:6]) for x in probabilities]

    # out put the result
    teamData = pd.DataFrame(np.array(win_prob), columns = [1])
    teamData.to_csv('../result.csv')


if __name__ == '__main__':

    # load data
    win_rate = np.load('team_win_rate.npy')
    training_set, label_set = loadData('../matchDataTrain.csv', win_rate)
    classifier, pca = train(training_set, label_set)
    predict(classifier, win_rate, pca)