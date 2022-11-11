import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

# 数据读取与拼接
'''
train.csv:包含全量数据集的70%（dataNoLabel是训练集的一部分，选手可以自己决定是否使用）
test.csv:包含全量数据集的30%
位置类特特征：基于联通基站产生的用户信令数据；
互联网类特征：基于联通用户上网产生的上网行为数据；
通话类特征：基于联通用户日常通话、短信产生的数据
'''
train = pd.read_csv(r'D:\桌面\乱七八糟\python实验\dataTrain.csv')
test = pd.read_csv(r'D:\桌面\乱七八糟\python实验\dataA.csv')
data = pd.concat([train, test]).reset_index(drop=True)
data['f3'] = data['f3'].map({'low': 0, 'mid': 1, 'high': 2})

# 暴力Feature 位置
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for i in range(len(loc_f)):
    for j in range(i + 1, len(loc_f)):
        data[f'{loc_f[i]}+{loc_f[j]}'] = data[loc_f[i]] + data[loc_f[j]]
        data[f'{loc_f[i]}-{loc_f[j]}'] = data[loc_f[i]] - data[loc_f[j]]
        data[f'{loc_f[i]}*{loc_f[j]}'] = data[loc_f[i]] * data[loc_f[j]]
        data[f'{loc_f[i]}/{loc_f[j]}'] = data[loc_f[i]] / data[loc_f[j]]

# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for i in range(len(com_f)):
    for j in range(i + 1, len(com_f)):
        data[f'{com_f[i]}+{com_f[j]}'] = data[com_f[i]] + data[com_f[j]]
        data[f'{com_f[i]}-{com_f[j]}'] = data[com_f[i]] - data[com_f[j]]
        data[f'{com_f[i]}*{com_f[j]}'] = data[com_f[i]] * data[com_f[j]]
        data[f'{com_f[i]}/{com_f[j]}'] = data[com_f[i]] / data[com_f[j]]

# 训练测试分离
train = data[~data['label'].isna()].reset_index(drop=True)
train = train[:50000]
test = data[data['label'].isna()].reset_index(drop=True)
test = test[:50000]

#数据处理，五折交叉验证
#为了防止数据不均匀，提高模型的精确度，减轻过拟合
features = [i for i in train.columns if i not in ['label', 'id']]#对数据进行特征选择
y = train['label']
KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
params = {
    'objective': 'binary',#目标函数 二分类
    'boosting_type': 'gbdt',#设置提升类型，传统的梯度提升决策树
    'metric': 'auc',#评估函数
    'n_jobs': 30,
    'learning_rate': 0.04,#学习速率
    'num_leaves': 2 ** 6,#叶子节点数
    'max_depth': 8,#树的最大深度，模型过拟合时，可优先考虑降低该数值
    'tree_learner': 'serial',
    'colsample_bytree': 0.82,
    'subsample_freq': 1,
    'subsample': 0.8,
    'num_boost_round': 5000,#boosting的迭代次数
    'max_bin': 255,# 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
    'verbose': -1,
    'seed': 2021,
    'bagging_seed': 2021,#bagging的随机种子数
    'feature_fraction_seed': 2021,#feature_fraction的随机种子数
    'early_stopping_rounds': 100,#如果一个验证集的度量在early_stopping_round 循环中没有提升，则停止训练

}

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros((len(test)))

# 模型训练：加入提前停止功能
for fold_, (trn_idx, val_idx) in enumerate(KF.split(train.values, y.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y.iloc[trn_idx])#训练集
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=y.iloc[val_idx])#测试集
    num_round = 3000#设置迭代次数为3000
    clf = lgb.train(
        params,
        trn_data,
        num_round,
        valid_sets=[trn_data, val_data],
        verbose_eval=100,#设置earlystop为100，如果迭代100次，loss没有发生变化则终止迭代
        early_stopping_rounds=50,
    )

    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(test[features], num_iteration=clf.best_iteration) / 5
    feat_imp_df['imp'] += clf.feature_importance() / 5

print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))#分类模型评估2/F1=1/R+1/P
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))#精确率 真正正确的占所有预测为正的比例。TP/(TP+FP)
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))#召回率 真正正确的占所有实际为正的比例。TP/(TP+FN)

# 提交结果
test['label'] = predictions_lgb
test[['id', 'label']].to_csv('_sub.csv', index=False)

