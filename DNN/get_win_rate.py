import pandas as pd
import numpy as np

data = pd.read_csv('../seedcupTask/matchDataTrain.csv')
result = np.zeros((208,3))
all_game = np.zeros((208,3))
value = data.values
for i in range(2,len(value)):
    away = int(value[i][0])
    home = int(value[i][1])
    str = value[i][4]
    str = str.split(':')
    away_score = int(str[0])
    home_score = int(str[1])
    all_game[home][0] += 1
    all_game[away][1] += 1
    all_game[home][2] += 1
    all_game[away][2] += 1
    if away_score > home_score:
        result[away][1] += 1
        result[away][2] += 1
    else:
        result[home][0] += 1
        result[home][2] += 1
for i in range(208):
    for j in range(3):
        if all_game[i][j]:
            result[i][j] = result[i][j]/all_game[i][j]
        else:
            result[i][j] = None

team_win_rate = pd.DataFrame(result)
team_win_rate.to_csv('team_win_rate.csv')
np.save('team_win_rate.npy', result)
