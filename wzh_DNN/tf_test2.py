#!/usr/bin/env python
# coding=utf-8

import os
import urllib
import tensorflow as tf
import numpy as np

def loadData(data_file, team_data):
    # load match information
    with open(data_file, 'r') as fp:
        fp.readline()
        lines = [line.split(',') for line in fp.readlines()]

    # data processing
    data_set = []
    label_set = []
    for line in lines:
        newdata = list(team_data[int(line[0])]) + list(team_data[int(line[1])])
        index2 = line[2].find('胜')
        index3 = line[3].find('胜')
        newdata += [int(line[2][:index2]), int(line[2][index2+1:-1]),
                    int(line[3][:index3]), int(line[3][index3+1:-1])]
        data_set.append(newdata)

    if len(lines[0]) == 5:
        for line in lines:
            score_data = line[4].split(':')
            diff = int(score_data[0]) - int(score_data[1])
            if diff > 0:
                newlabel = 0 if diff < 4 else (1 if diff < 11 else 2)
            else:
                newlabel = 3 if diff > -4 else (4 if diff > -11 else 5)
            label_set.append(newlabel)

    # print(data_set[0])
    return data_set, label_set


def train(training_set, label_set):
    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column(
        "x", shape=[len(training_set[0])])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[40,60],
                                            n_classes=6,)
                                            # model_dir="./seedcup_model")

    # Define the training inputs    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set[:6000])},
        y=np.array(label_set[:6000]),
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=65000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set[:6000])},
        y=np.array(label_set[:6000]).T,
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTrain Accuracy: {0:f}\n".format(accuracy_score))

    # Define the test inputs
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set[6000:])},
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    predictions = list(classifier.predict(input_fn=predict_input_fn))
    classes = [p['classes'] for p in predictions]

    test_label_set = label_set[6000:]
    length = len(classes)
    right = 0
    for i in range(length):
        #print(classes[i][0], test_label_set[i])
        if (int(classes[i][0].decode('utf8')) - 2.5) * (test_label_set[i] - 2.5) >= 0:
            right += 1

    print(
        "New Samples, Accuracy: {}\n"
        .format(right / length))
    return classifier


# def predict(classifier):
#     # Classify two new flower samples.
#     new_samples = 
#     predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": new_samples},
#         num_epochs=1,
#         shuffle=False)

#     predictions = list(classifier.predict(input_fn=predict_input_fn))
#     probabilities = [p['probabilities'] for p in predictions]

#     print(
#         "New Samples, probabilities: {}\n"
#         .format(probabilities))


if __name__ == '__main__':
    team_data = np.load('team_data.npy')
    training_set, label_set = loadData(
        '../matchDataTrain.csv', team_data)
    classifier = train(training_set, label_set)
# /tmp/tmpxjtir9ne



'''
40 60 50000 0.7721
40 60 60000 0.773
40 60 70000 0.74
40 60 65000 0.7952
'''