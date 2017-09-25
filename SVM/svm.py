#!/usr/bin/env python
# coding=utf-8

from sklearn import svm, preprocessing
import numpy as np


def loadData(data_file, team_data):
    # load match information
    with open(data_file, 'r') as fp:
        fp.readline()
        lines = [line.split(',') for line in fp.readlines()]

    # get data
    data_set = []
    label_set = []
    for line in lines:
        # expand team information
        newdata = list(team_data[int(line[0])]) + list(team_data[int(line[1])])
        # get win and lose
        index2 = line[2].find('胜')
        index3 = line[3].find('胜')
        newdata += [int(line[2][:index2]), int(line[2][index2+1:-1]),
                    int(line[3][:index3]), int(line[3][index3+1:-1])]
        data_set.append(newdata)

    if len(lines[0]) == 5:  # if there is match result
        for line in lines:
            score_data = line[4].split(':')
            diff = int(score_data[0]) - int(score_data[1])
            if diff > 0:
                # if (0, 4) 0; if [4, 11) 1; if [11, ...) 2
                newlabel = 0 if diff < 4 else (1 if diff < 11 else 2)
            else:
                # if (-4, 0] 0; if (-11, -4] 4; if (..., -11] 5
                newlabel = 3 if diff > -4 else (4 if diff > -11 else 5)
            label_set.append(newlabel)

    # print(data_set[0])
    return data_set, label_set


def train(training_set, label_set):
    std = preprocessing.StandardScaler()
    training = std.fit_transform(training_set[:6000])
    test = std.fit_transform(training_set[6000:])
    clf = svm.LinearSVC(max_iter=2000, C=1)
    clf.fit(training, label_set[:6000])

    print('train accuracy: ', clf.score(training, label_set[:6000]))

    classes = clf.predict(test)

    test_label_set = label_set[6000:]
    length = len(classes)
    right = 0
    for i in range(length):
        if (classes[i] - 2.5) * (test_label_set[i] - 2.5) > 0:
            right += 1
    print('Accuracy: {}'.format(right / length))


def predict(classifier):
    pass


if __name__ == '__main__':
    team_data = np.load('../DNN/team_data.npy')
    training_set, label_set = loadData(
        '../matchDataTrain.csv', team_data)
    classifier = train(training_set, label_set)
# /tmp/tmpxjtir9ne
