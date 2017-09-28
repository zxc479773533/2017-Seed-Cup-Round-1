#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib as plt

# 打开文件
linenum = 208
MVPnum = 10
feature_number_of_every_member = 0

data = pd.read_csv('../seedcupTask/teamData.csv')
row = range(linenum)  # 设置csv的行，一共208个队伍
col = []  # 设置列

# 初始化列字符串
for i in range(MVPnum):
    # col.append('each')
    # col.append('player'+str(i)+'_time')
    # col.append('player'+str(i)+'_shot')
    # # col.append('player'+str(i)+'_three')
    # # col.append('player'+str(i)+'_one')
    # col.append('player'+str(i)+'_score')
    # col.append('player'+str(i)+'_hit')
    pass
# col.append('rebounds')
col.append('total_score')
# col.append('MVPscore')
# col.append('member_numbers')
# col.append('assidts')
# col.append('steals')
# col.append('cap')
# col.append('foul')
# col.append('mistakes')

colnum = len(col)

# linenum raw 61 column 生成所有整理好信息的csv
team_np = np.zeros((linenum, colnum))
# 处理数据
for i in range(linenum):
    team = []  # 存储返回列表信息
    # 拿到队伍信息
    one_team = data[data['队名'] == i]
    # 按照出场时间排序
    one_team = one_team.sort_values(by='投篮出手次数', ascending=False)
    # 信息化为矩阵方便操作
    values = one_team.values
    member_numbers = values.shape[0]
    # team_np[i][MVPnum*feature_number_of_every_member+2] = values[0][22]
    # team_np[i][(MVPnum+1)*feature_number_of_every_member+1] = member_numbers
    for j in range(MVPnum):
        # if values[j][7]:
        #     team_np[i][j*feature_number_of_every_member] = values[j][6]/values[j][7]
        # else:
        #     team_np[i][j*feature_number_of_every_member] = 0
        # if values[j][10]:
        #     team_np[i][j*feature_number_of_every_member+1] = (values[j][4]/48)*values[j][9]/values[j][10]
        # else:
        #     team_np[i][j*feature_number_of_every_member+1] = 0
        # if values[j][13]:
        #     team_np[i][j*feature_number_of_every_member+2] = (values[j][4]/48)*values[j][12]/values[j][13]
        # else:
        #     team_np[i][j*feature_number_of_every_member+2] = 0
        # team_np[i][j*feature_number_of_every_member+1] = values[j][22]
        # team_np[i][j*feature_number_of_every_member+2] = values[j][6]
        # team_np[i][j] = (values[j][22]+values[j][14]+values[j][17]+values[j][18]+values[j][19])-(values[j][7]-values[j][6])-(values[j][10]-values[j][9])-(values[j][13]-values[j][12])-values[j][20] 
        # team_np[i][MVPnum*feature_number_of_every_member] += values[j][14]
        team_np[i][MVPnum*feature_number_of_every_member] += values[j][22]
        pass
    # total_time = 0
    # total_shot = 0
    # # total_three = 0
    # # total_one = 0
    # total_score = 0
    # total_hit = 0
    for j in range(MVPnum, member_numbers):
        # team_np[i][MVPnum*feature_number_of_every_member] += values[j][14]
        team_np[i][MVPnum*feature_number_of_every_member] += values[j][22]
        # total_time += values[j][4]
        # total_shot += values[j][6]/values[j][7] if values[j][7] else 0
        # # total_three += values[j][9]/values[j][10] if values[j][10] else 0
        # # total_one += values[j][12]/values[j][13] if values[j][13] else 0
        # total_score += values[j][22]
        # total_hit += values[j][6]
        pass
    # team_np[i][MVPnum*feature_number_of_every_member] = total_shot*(member_numbers-MVPnum)
    # # team_np[i][MVPnum*feature_number_of_every_member+1] = total_three*total_time/48/(member_numbers-MVPnum)
    # # team_np[i][MVPnum*feature_number_of_every_member+2] = total_one*total_time/48/(member_numbers-MVPnum)
    # team_np[i][MVPnum*feature_number_of_every_member+1] = total_score/(member_numbers-MVPnum)
    # team_np[i][MVPnum*feature_number_of_every_member+2] = total_hit/(member_numbers-MVPnum)

print(len(team_np))
print(len(team_np[1]))

team_data = pd.DataFrame(team_np, index=row, columns=col)
team_data.to_csv('team_data.csv')
np.save('team_data.npy', team_np)
