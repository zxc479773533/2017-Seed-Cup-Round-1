#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


def loadData(data_file, team_data, win_rate, all_history=None):
    # load match information
    with open(data_file, 'r', encoding='utf-8') as fp:
        fp.readline()
        lines = [line.split(',') for line in fp.readlines()]

    # get data
    data_set = []
    label_set = []
    history = None
    for line in lines:
        # expand team information
        newdata = []
        # get win and lose
        line[2] = line[2].strip()
        line[3] = line[3].strip()
        index2 = line[2].find('胜')
        index3 = line[3].find('胜')
        # newdata += [int(line[2][:index2]), int(line[2][index2+1:-1]),
        #             int(line[3][:index3]), int(line[3][index3+1:-1])]
        guest_id = int(line[0])
        home_id = int(line[1])
        newdata.append((win_rate[guest_id][1] - win_rate[home_id][0]) * 30)
        newdata.append((win_rate[guest_id][2] - win_rate[home_id][2]) * 30)
        # guest = list(team_data[int(line[0])])
        # home = list(team_data[int(line[1])])
        # newdata += [guest[i] - home[i] for i in range(len(home))]
        data_set.append(newdata)

    if len(lines[0]) == 5:  # if there are match results
        history = np.zeros((208, 208))
        for line in lines:
            score_data = line[4].split(':')
            diff = int(score_data[0]) - int(score_data[1])
            if diff > 0:
                # if (0, 4) 0; if [4, 11) 1; if [11, ...) 2
                newlabel = 1 if diff < 6 else (2 if diff < 16 else 3)
                history[int(line[0].strip())][int(line[1].strip())] += newlabel**2
                newlabel = 0
            else:
                # if (-4, 0] 3; if (-11, -4] 4; if (..., -11] 5
                newlabel = 1 if diff > -6 else (2 if diff > -16 else 3)
                history[int(line[0].strip())][int(line[1].strip())] -= newlabel**2
                newlabel = 1
            label_set.append(newlabel)
    else:
        history = []
        for line in lines:
            history.append(all_history[int(line[0].strip())] \
                [int(line[1].strip())] * 0.01 - all_history[
                int(line[1].strip())][int(line[0].strip())] * 0.005)

    # print(data_set[0])
    return data_set, label_set, history


def train_all(training_set, label_set):
    n = 5
    print(n)
    pca = PCA(n_components=n)
    training_set = pca.fit_transform(training_set)
    print(pca.explained_variance_ratio_)
    # return

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column(
        "x", shape=[len(training_set[0])])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    hidden_units = [6]
    print(hidden_units)
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=hidden_units,
                                            n_classes=2,)
    # model_dir="./seedcup_model")

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set)},
        y=np.array(label_set),
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=2000)
    return classifier, pca


def train(training_set, label_set):
    n = None
    print(n)
    pca = PCA(n_components=n)
    training_set = pca.fit_transform(training_set)
    print(pca.explained_variance_ratio_)
    # return

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column(
        "x", shape=[len(training_set[0])])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    hidden_units = [4]
    print(hidden_units)
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=hidden_units,
                                            n_classes=2,)
    # model_dir="./seedcup_model")

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set[2000:])},
        y=np.array(label_set[2000:]),
        num_epochs=None,
        shuffle=True)

    # Train model.
    # classifier.train(input_fn=train_input_fn, steps=25000)
    for k in range(10):
        classifier.train(input_fn=train_input_fn, steps=1000)

        # Define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(training_set[2000:])},
            y=np.array(label_set[2000:]).T,
            num_epochs=1,
            shuffle=False)

        # Evaluate accuracy.
        accuracy_score = classifier.evaluate(
            input_fn=test_input_fn)["accuracy"]

        print("\nTrain Accuracy: {0:f}\n".format(accuracy_score))

        # Define the test inputs
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(training_set[:2000])},
            num_epochs=1,
            shuffle=False)

        # Evaluate accuracy.
        predictions = list(classifier.predict(input_fn=predict_input_fn))
        classes = [p['classes'] for p in predictions]
        probabilities = [p['probabilities'] for p in predictions]
        win_prob = [x[1] for x in probabilities]

        test_label_set = label_set[:2000]
        length = len(classes)
        right = 0
        for i in range(length):
            if (int(classes[i][0].decode('utf8')) - 0.5) * \
             (test_label_set[i] - 0.5) > 0:
                right += 1

        test_label_set = [0 if x < 0.5 else 1 for x in test_label_set]

        print('train steps number: ', (k+1)*500)
        print("New Samples, Accuracy: ", right / length)
        print('AUC score: ', roc_auc_score(test_label_set, win_prob))
        print()
    return classifier, pca


def predict(classifier, team_data, win_rate, all_history, pca):
    # Classify two new flower samples.
    new_samples, labels, history = loadData(
        '../seedcupTask/matchDataTest.csv', team_data, win_rate, all_history)
    new_samples = pca.transform(new_samples)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(new_samples)},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    probabilities = [p['probabilities'] for p in predictions]
    win_prob = [x[1] for x in probabilities]
    print(list(zip(win_prob, history)))
    for i in range(len(win_prob)):
        win_prob[i] = max(win_prob[i] - history[i], 0.00001)
        win_prob[i] = min(win_prob[i], 0.99999)

    team_data = pd.DataFrame(np.array(win_prob), columns=[1])
    team_data.to_csv('result.csv')


if __name__ == '__main__':
    team_data = np.load('team_data.npy')
    win_rate = np.load('team_win_rate.npy')
    training_set, label_set, all_history = loadData(
        '../2017-Seed-Cup-Round-1/matchDataTrain.csv', team_data, win_rate)
    classifier, pca = train(training_set, label_set)
    predict(classifier, team_data, win_rate, all_history, pca)
    # np.save('history.npy', history)


# 5 [5] 38000 with winrate
# 4 features without team data [5] 8000 10000
