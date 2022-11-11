# 公榜分数0.92860690761
#引入pandas
import os
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import numpy as np
#引入微软的机器学习库
import lightgbm as lgb
import time

import pydotplus as pydotplus

os.environ["PATH"] += os.pathsep + 'D:\Graphviz\bin'

from matplotlib import pyplot as plt
#labelEncoder 对数据集进行编码
from sklearn.preprocessing import LabelEncoder

#roc_auc_score返回曲线下面积  输入是两个数组
#用于计算auc的值
from sklearn.metrics import roc_auc_score as auc, roc_curve
#Kfold用于生成交叉验证数据集,StratifiedFold是在fold基础上加入分层抽样的思想,使训练集和测试集有相同的分布,在算法上
#表现为需要同时输入数据和标签 输入数据表现为数组 split得出生成器
from sklearn.model_selection import StratifiedKFold

#用pandas的数据读取工具读取csv文件 (一次性读入)
df_train = pd.read_csv('data/data170933/train.csv')
df_test = pd.read_csv('data/data170933/evaluation_public.csv')


###特征构造
#纵向拼接两个表
df = pd.concat([df_train, df_test])

#转换为pandas中datetime类型的时间类型数据。
df['op_datetime'] = pd.to_datetime(df['op_datetime'])
df['hour'] = df['op_datetime'].dt.hour
df['dayofweek'] = df['op_datetime'].dt.dayofweek

#将数据表按照用户名和操作时间进行排序,并且重置索引
df = df.sort_values(by=['user_name', 'op_datetime']).reset_index(drop=True)

#df.info()
# #显示所有列
# pd.set_option('display.max_columns',None)
# #显示所有行
# pd.set_option('display.max_rows',None)
# print(df)

 #新建一个ts列存放op_datetime原来是datatime类型列强类型转换成int64类型
df['ts'] = df['op_datetime'].values.astype(np.int64) // 10 ** 9
#df.info()
#print(df['ts'])

#移动索引轴
df['ts1'] = df.groupby('user_name')['ts'].shift(1)

df['ts2'] = df.groupby('user_name')['ts'].shift(2)

df['ts_diff1'] = df['ts1'] - df['ts']

df['ts_diff2'] = df['ts2'] - df['ts']

df['hour_sin'] = np.sin(df['hour']/24*2*np.pi)
df['hour_cos'] = np.cos(df['hour']/24*2*np.pi)
#print(df['hour_cos'])
LABEL = 'is_risk'
df.to_csv('mm7.csv', index=False, encoding="utf-8")
cat_f = ['user_name', 'department', 'ip_transform', 'device_num_transform', 'browser_version', 'browser',
         'os_type', 'os_version', 'ip_type', 'op_city', 'log_system_transform', 'url',]

traindata = df[~df['is_risk'].isnull()].reset_index(drop=True)
data_Y_test = traindata['is_risk'].values[40000:].astype(int).reshape(-1, 1)

#归一化数据
for f in cat_f:
    le = LabelEncoder()
    df[f] = le.fit_transform(df[f])
    # 先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的）
    # df[f+'_ts_diff_mean'] = df.groupby([f])['ts_diff1'].transform('mean') #提取词频生成词频矩阵
    # df[f+'_ts_diff_std'] = df.groupby([f])['ts_diff1'].transform('std')
# df.to_csv('mm8.csv', index=False, encoding="utf-8")
#print(df['user_name_ts_diff_std'])
#存储LABEL列中为空值的行
df_train = df[df[LABEL].notna()].reset_index(drop=True)

#df_train.to_csv('df_train.csv', index=False, encoding="utf-8")
#存储LABEL列中不为空值的行
df_test = df[df[LABEL].isna()].reset_index(drop=True)
# print("之后")
# print(df_train[LABEL].shape)
# print(df_test[LABEL].shape)

#df_test.to_csv('df_test.csv', index=False, encoding="utf-8")
#与LABEL分开且去掉用不到的
feats = [f for f in df_test if f not in [LABEL, 'id',
                                         'op_datetime', 'op_month', 'ts', 'ts1', 'ts2']]
#print(feats)
#输出训练集和测试集的行数列数
#print(df_train[feats].shape, df_test[feats].shape)

#机器学习模型
#配置训练模型的各项参数
params = {

    # default=0.1, type=double, alias=shrinkage_rate
    'learning_rate': 0.0015,
    'boosting_type': 'gbdt',

    # default=regression，任务类型
    'objective': 'binary',
    'metric': 'auc',
    #叶子节点数量
    'num_leaves': 64,
    # default=1, type=int, alias=verbose  |  日志冗长度，[详细信息]代表是否输出 < 0: Fatal, = 0: Error (Warn), > 0: Info
    'verbose': -1,
    'seed': 2222,
    'n_jobs': -1,

    'feature_fraction': 0.8,
#default=1.0, type=double, 0.0 < feature_fraction < 1.0, alias=sub_feature,                                                                        #colsample_bytree
    # 如果 feature_fraction 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征. 例如, 如果设置为 0.8, 将会在每棵树训练之前选择 80% 的特征
    # 可以用来加速训练
    # 可以用来处理过拟合
    'bagging_fraction': 0.9,
# default=1.0, type=double, 0.0 < bagging_fraction < 1.0, alias=sub_row, subsample
    # 类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
    # 可以用来加速训练
    # 可以用来处理过拟合
    # Note: 为了启用 bagging, bagging_freq 应该设置为非零值
    'bagging_freq': 4,
    # 'min_child_weight': 10,
}

#训练数据
fold_num = 10 #数据分割点 前几份作为测试集 后面作为训练集 训练次数
seeds = [2222]
#设置一个和训练集形状一样的全是零的数组
oof = np.zeros(len(df_train))
importance = 0
#定义一个空表
pred_y = pd.DataFrame()
score = []
axx = 0.
inputs =[]
outputs = []
from IPython.display import Image
from graphviz import dot
for seed in seeds:
      #交叉验证,打乱数据分层抽样
    kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    # kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
      #(遍历索引，遍历元素enumerate相当于in range(len())

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL])):
        print('-----------', fold)
        print('train_idx: %s | val_idx: %s' % (train_idx, val_idx))
        #数据集分割
        train = lgb.Dataset(df_train.loc[train_idx, feats],
                            df_train.loc[train_idx, LABEL])
        val = lgb.Dataset(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])
         #模型组网
        model = lgb.train(params, train, valid_sets=[val], num_boost_round=20000,
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(2000)])

        oof[val_idx] += model.predict(df_train.loc[val_idx, feats]) / len(seeds)
        pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(df_test[feats])
        importance += model.feature_importance(importance_type='gain') / fold_num
        a =auc(df_train.loc[val_idx, LABEL], model.predict(df_train.loc[val_idx, feats]))
        score.append(a)
        # dot_data =lgb.create_tree_digraph(model, tree_index=0, encoding='UTF-8')
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # Image(graph.create_png())


feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
#输出前30行按importance降序排序后表 一共有43行
print(feats_importance.sort_values('importance', ascending=False)[:30])
print(feats_importance.shape)
df_train['oof'] = oof
# #输出均值和标准差
print(np.mean(score), np.std(score))
#
score = np.mean(score)

# #求列平均值
df_test[LABEL] = pred_y.values.mean()


#
# #print(df_test)
#输出结果
sub = pd.read_csv('data/submit_sample.csv')
sub['ret'] = df_test[LABEL].values
sub.columns = ['id', LABEL]
sub.to_csv(time.strftime('ans/sumission_%Y%m%d%H%M_')+'%.5f.csv'%score, index=False)