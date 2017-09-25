#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib as plt
# 打开文件
data = pd.read_csv('../teamData.csv')
row = range(0, 208)  # 设置csv的行，一共208个队伍
col = []  # 设置列
# 初始化列字符串
for i in range(11):
    col.append('player'+str(i)+'_time')
    col.append('player'+str(i)+'_shot')
    col.append('player'+str(i)+'_three')
    col.append('player'+str(i)+'_one')
    col.append('player'+str(i)+'_score')
col.append('rebounds')
col.append('assidts')
col.append('steals')
col.append('cap')
col.append('foul')
col.append('mistakes')
# 208 raw 61 column 生成所有整理好信息的csv
team_np = np.zeros((208, 61))
# 处理数据
for i in range(208):
    team = []  # 存储返回列表信息
    # 拿到队伍信息
    one_team = data[(data['队名'] > i-1) & (data['队名'] < i+1)]
    # 按照出场时间排序
    one_team = one_team.sort_values(by='上场时间', ascending=False)
    # 信息化为矩阵方便操作
    values = one_team.values
    team_numbers = values.shape[0]
    for j in range(10):
        team_np[i][j*5+0] = values[j][4]/48
        team_np[i][j*5+1] = (values[j][4]/48)*values[j][6]
        team_np[i][j*5+2] = (values[j][4]/48)*values[j][9]
        team_np[i][j*5+3] = (values[j][4]/48)*values[j][12]
        team_np[i][j*5+4] = values[j][22]
        team_np[i][55] -= values[j][20] #失误
        team_np[i][56] += values[j][14] #篮板
        team_np[i][57] += values[j][17] #助攻
        team_np[i][58] += values[j][18] #抢断
        team_np[i][59] += values[j][19] #盖帽
        team_np[i][60] += values[j][22] #得分
    total_time = 0
    total_shot = 0
    total_three = 0
    total_one = 0
    total_score = 0
    for j in range(10, team_numbers):
        team_np[i][55] += values[j][14]
        team_np[i][56] += values[j][17]
        team_np[i][57] += values[j][18]
        team_np[i][58] += values[j][19]
        team_np[i][59] += values[j][20]
        team_np[i][60] += values[j][21]
        total_time += values[j][4]
        total_shot += values[j][6]
        total_three += values[j][9]
        total_one += values[j][12]
        total_score += values[j][22]
    team_np[i][50] = total_time/48/(team_numbers-10)
    team_np[i][51] = total_shot*total_time/48/(team_numbers-10)
    team_np[i][52] = total_three*total_time/48/(team_numbers-10)
    team_np[i][53] = total_one*total_time/48/(team_numbers-10)
    team_np[i][54] = total_score/(team_numbers-10)
team_data = pd.DataFrame(team_np, index=row, columns=col)
print(len(team_np))
team_data.to_csv('team_data.csv')
np.save('team_data.npy', team_np)
