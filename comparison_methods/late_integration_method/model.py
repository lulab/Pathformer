import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score, f1_score, accuracy_score
from scipy.stats import rankdata
import argparse, sys, os, errno
import logging
import warnings
import joblib
import random
warnings.filterwarnings("ignore")




def plot_roc(prob, label):
    fpr, tpr, _ = roc_curve(label, prob)
    precision, recall, _ = precision_recall_curve(label, prob)  # recall: Identical to sensitivity,TPR; precision: PPV
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    # print (fpr, tpr,ppv)
    return roc_auc, fpr, tpr, precision, recall


def get_roc(y, pro):
    label=np.array(y)
    pre_label = (pro>0.5).astype(int)
    auc=roc_auc_score(label, np.array(pro))
    acc = accuracy_score(label, pre_label)
    f1_macro = f1_score(label, pre_label, average='macro')
    f1_weighted = f1_score(label, pre_label, average='weighted')
    return acc, auc,f1_weighted, f1_macro

def plot_roc_multi(prob, label):
    pre_label = prob.argmax(axis=1)
    acc = accuracy_score(label, pre_label)
    auc_macro_ovr = roc_auc_score(label, prob, average='macro', multi_class='ovr')
    auc_macro_ovo = roc_auc_score(label, prob, average='macro', multi_class='ovo')
    auc_weighted_ovr = roc_auc_score(label, prob, average='weighted', multi_class='ovr')
    auc_weighted_ovo = roc_auc_score(label, prob, average='weighted', multi_class='ovo')
    f1_macro = f1_score(label, pre_label, average='macro')
    f1_weighted = f1_score(label, pre_label, average='weighted')
    return acc, auc_weighted_ovr, auc_weighted_ovo, auc_macro_ovr, auc_macro_ovo, f1_weighted, f1_macro


def clf_select(name, pds):
    if name == 'DT':
        clf = DecisionTreeClassifier(max_depth=100, min_samples_leaf=5, criterion='gini')
    elif name == 'DT_cv':
        tree_para = {'max_depth': [50, 100, 200, 500, 1000]}
        clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=pds, n_jobs=5, scoring='f1_macro')
    elif name == 'SVM':
        clf = SVC(kernel='rbf', probability=True, C=1)
    elif name == 'SVM_cv':
        tree_para = {'C': [0.01, 0.1, 1, 10, 100]}
        clf = GridSearchCV(SVC(kernel='rbf', probability=True), tree_para, cv=pds, n_jobs=5, scoring='f1_macro')
    elif name == 'RF':
        clf = RandomForestClassifier(n_estimators=50, max_depth=1000,ccp_alpha=0.1)
    elif name == 'RF_cv':
        tree_para = {'n_estimators': [10,50, 100, 200, 500], 'max_depth': [10, 50,100,200, 500]}
        clf = GridSearchCV(RandomForestClassifier(), tree_para, cv=pds, n_jobs=5, scoring='f1_macro')
    elif name == 'LR':
        clf = LogisticRegression(penalty='l2', solver='liblinear', C=1)
    elif name == 'LR_cv':
        tree_para = {'C': [ 0.001, 0.1, 1, 10, 100]}
        clf = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear'), tree_para, cv=pds,n_jobs=5, scoring='f1_macro')
    elif name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=10, weights='distance', leaf_size=10)
    elif name == 'KNN_cv':
        tree_para = {'n_neighbors': [5, 10, 20, 50]}
        clf = GridSearchCV(KNeighborsClassifier(weights='distance'), tree_para, cv=pds, n_jobs=5, scoring='f1_macro')
    elif name == 'NN':
        clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=200)
    elif name == 'LGBoost':
        clf = LGBMClassifier(num_leaves=5, n_estimators=100)
    elif name == 'LGBoost_cv':
        tree_para = {'max_depth': [5,10,50,100,500,1000], 'n_estimators': [100,500,1000],'num_leaves':[20,30,50,100]}
        clf = GridSearchCV(LGBMClassifier(learning_rate=0.1), tree_para, cv=pds, n_jobs=5, scoring='f1_macro')
    elif name == 'XGBoost':
        clf = xgb.XGBClassifier(learning_rate=0.5, n_estimators=500, max_depth=1000, min_child_weight=3,
                                gamma=3,  # 惩罚项中叶子结点个数前的参数
                                subsample=0.7,  # 随机选择80%样本建立决策树
                                objective='binary:logistic',  # 指定损失函数
                                # scale_pos_weight=1,  # 解决样本个数不平衡的问题
                                nthread=5
                                )
    elif name == 'XGBoost_cv':
        tree_para = {'max_depth': [10, 50, 100, 200, 500], 'n_estimators': [50, 100, 200, 500]}
        clf = GridSearchCV(xgb.XGBClassifier(learning_rate=0.5, min_child_weight=3, gamma=3, subsample=0.7,
                                             objective='binary:logistic',
                                             scale_pos_weight=1, nthread=5), tree_para, cv=pds,n_jobs=5, scoring='f1_macro')
    return clf


def clf_select_multi(name, pds):
    if name == 'DT':
        clf = DecisionTreeClassifier(max_depth=100, min_samples_leaf=5, criterion='gini')
    elif name == 'DT_cv':
        tree_para = {'max_depth': [50, 100, 200, 500, 1000]}
        clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=pds, n_jobs=5, scoring='f1_macro')
    elif name == 'SVM':
        clf = SVC(kernel='rbf', probability=True, C=1)
    elif name == 'SVM_cv':
        tree_para = {'C': [0.01, 0.1, 1, 10, 100]}
        clf = GridSearchCV(SVC(kernel='rbf', probability=True), tree_para, cv=pds, n_jobs=5,scoring='f1_macro')
    elif name == 'RF':
        clf = RandomForestClassifier(n_estimators=100, max_depth=100)
    elif name == 'RF_cv':
        tree_para = {'n_estimators': [10,50, 100, 200, 500], 'max_depth': [10, 50,100,200, 500]}
        clf = GridSearchCV(RandomForestClassifier(), tree_para, cv=pds, n_jobs=5, scoring='f1_macro')
    elif name == 'LR':
        clf = LogisticRegression(penalty='l2', solver='liblinear', C=1)
    elif name == 'LR_cv':
        tree_para = {'C': [ 0.001, 0.1, 1, 10, 100]}
        clf = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear'), tree_para, cv=pds,n_jobs=5, scoring='f1_macro')
    elif name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=10)
    elif name == 'KNN_cv':
        tree_para = {'n_neighbors': [5, 10, 20, 50]}
        clf = GridSearchCV(KNeighborsClassifier(weights='distance'), tree_para, cv=pds, n_jobs=5, scoring='f1_macro')
    elif name == 'NN':
        clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=200)
    elif name == 'LGBoost':
        clf = LGBMClassifier( num_leaves=5, n_estimators=100)
    elif name == 'LGBoost_cv':
        tree_para = {'max_depth': [5,10,50,100,500,1000], 'n_estimators': [100,500,1000],'num_leaves':[20,30,50,100]}
        clf = GridSearchCV(LGBMClassifier(learning_rate=0.1), tree_para, cv=pds, n_jobs=5, scoring='f1_macro')
    elif name == 'XGBoost':
        clf = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=100, min_child_weight=1,
                                gamma=0.,  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,  # 随机选择80%样本建立决策树
                                objective='multi:softprob',  # 指定损失函数
                                nthread=5
                                )
    elif name == 'XGBoost_cv':
        tree_para = {'max_depth': [10, 50, 100, 200, 500], 'n_estimators': [50, 100, 200, 500]}
        clf = GridSearchCV(xgb.XGBClassifier(learning_rate=0.5, min_child_weight=3, gamma=3, subsample=0.7,
                                             objective='binary:logistic',
                                             scale_pos_weight=1, nthread=5), tree_para, cv=pds,n_jobs=5, scoring='f1_macro')
    return clf


def main_model(data_path, label_path, feature_name, dataset, method, num, save_path, type):
    data = pd.read_csv(data_path + '/data_' + type + '.txt', sep='\t')
    data = data.rename(columns={data.columns[0]: 'feature'})
    data = data.fillna(0)
    label = pd.read_csv(label_path, sep='\t')
    sample_discovery = label.loc[label['dataset_' + str(dataset)] != 'test', 'sample_id']
    sample_validation = label.loc[label['dataset_' + str(dataset)] == 'test', 'sample_id']
    label = label.set_index('sample_id')
    y_discovery = np.array(label.loc[sample_discovery, 'y'])
    y_discovery = y_discovery.astype('int')
    y_validation = np.array(label.loc[sample_validation, 'y'])
    y_validation = y_validation.astype('int')
    X_discovery = np.array(data.loc[:, sample_discovery]).T
    X_validation = np.array(data.loc[:, sample_validation]).T
    result_predict_discovery = pd.DataFrame(columns={'sample_id', 'label'})
    result_predict_discovery['sample_id'] = sample_discovery
    result_predict_discovery['label'] = list(label.loc[sample_discovery, 'y'])
    result_predict_validation = pd.DataFrame(columns={'sample_id', 'label'})
    result_predict_validation['sample_id'] = sample_validation
    result_predict_validation['label'] = list(label.loc[sample_validation, 'y'])
    result_auc = pd.DataFrame(columns={'AUC_discovery', 'AUC_validation'})
    pds = 5
    for i in range(num):
        sample_weight_ = compute_sample_weight(class_weight='balanced', y=y_discovery)
        if len(set(label['y'])) > 2:
            clf = clf_select_multi(method, pds)
        else:
            clf = clf_select(method, pds)
        if (method == 'KNN') | (method == 'KNN_cv') | (method == 'NN'):
            clf.fit(X_discovery, y_discovery)
        else:
            clf.fit(X_discovery, y_discovery, sample_weight_)

        if len(set(label['y'])) > 2:
            predict_discovery = clf.predict_proba(X_discovery)
            predict_validation = clf.predict_proba(X_validation)
            y_discovery_index = list(set(y_discovery))
            y_discovery_index.sort()
            y_validation_index = list(set(y_validation))
            y_validation_index.sort()
            predict_discovery_ = predict_discovery[:, y_discovery_index] / predict_discovery[:, y_discovery_index].sum(
                axis=1, keepdims=1)
            predict_validation_ = predict_validation[:, y_validation_index] / predict_validation[:,
                                                                              y_validation_index].sum(axis=1,
                                                                                                      keepdims=1)
            predict_discovery_[np.isnan(predict_discovery_)] = 1 / predict_discovery_.shape[1]
            predict_validation_[np.isnan(predict_validation_)] = 1 / predict_validation_.shape[1]
            acc_discovery, auc_weighted_ovr_discovery, auc_weighted_ovo_discovery, auc_macro_ovr_discovery, auc_macro_ovo_discovery, f1_weighted_discovery, f1_macro_discovery = plot_roc_multi(
                predict_discovery_, rankdata(y_discovery, method='dense') - 1)
            acc_validation, auc_weighted_ovr_validation, auc_weighted_ovo_validation, auc_macro_ovr_validation, auc_macro_ovo_validation, f1_weighted_validation, f1_macro_validation = plot_roc_multi(
                predict_validation_, rankdata(y_validation, method='dense') - 1)
            for y_num in list(set(y_discovery)):
                result_predict_discovery['num_' + str(y_num) + '_' + str(i)] = predict_discovery[:, y_num]
                result_predict_validation['num_' + str(y_num) + '_' + str(i)] = predict_validation[:, y_num]
            result_auc.loc[i, 'acc_discovery'] = acc_discovery
            result_auc.loc[i, 'auc_weighted_ovr_discovery'] = auc_weighted_ovr_discovery
            result_auc.loc[i, 'auc_weighted_ovo_discovery'] = auc_weighted_ovo_discovery
            result_auc.loc[i, 'auc_macro_ovr_discovery'] = auc_macro_ovr_discovery
            result_auc.loc[i, 'auc_macro_ovo_discovery'] = auc_macro_ovo_discovery
            result_auc.loc[i, 'f1_weighted_discovery'] = f1_weighted_discovery
            result_auc.loc[i, 'f1_macro_discovery'] = f1_macro_discovery
            result_auc.loc[i, 'acc_validation'] = acc_validation
            result_auc.loc[i, 'auc_weighted_ovr_validation'] = auc_weighted_ovr_validation
            result_auc.loc[i, 'auc_weighted_ovo_validation'] = auc_weighted_ovo_validation
            result_auc.loc[i, 'auc_macro_ovr_validation'] = auc_macro_ovr_validation
            result_auc.loc[i, 'auc_macro_ovo_validation'] = auc_macro_ovo_validation
            result_auc.loc[i, 'f1_weighted_validation'] = f1_weighted_validation
            result_auc.loc[i, 'f1_macro_validation'] = f1_macro_validation
        else:
            predict_discovery = clf.predict_proba(X_discovery)
            roc_auc_discovery, fpr_discovery, tpr_discovery, precision_discovery, recall_discovery = plot_roc(
                predict_discovery[:, 1], y_discovery)
            predict_validation = clf.predict_proba(X_validation)
            roc_auc_validation, fpr_validation, tpr_validation, precision_validation, recall_validation = plot_roc(
                predict_validation[:, 1], y_validation)
            result_predict_discovery['num_' + str(i)] = predict_discovery[:, 1]
            result_predict_validation['num_' + str(i)] = predict_validation[:, 1]
            result_auc.loc[i, 'AUC_discovery'] = roc_auc_discovery
            result_auc.loc[i, 'AUC_validation'] = roc_auc_validation
    result_predict_discovery.to_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_discovery_' + type + '_' + feature_name + '.txt', sep='\t',index=False)
    result_predict_validation.to_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_validation_' + type + '_' + feature_name + '.txt', sep='\t',index=False)
    result_auc.to_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_AUC_' + type + '_' + feature_name + '.txt', sep='\t', index=False)


def merge(dataset, feature_name, method, num, save_path, label_path):
    label = pd.read_csv(label_path, sep='\t')
    result_predict_discovery_all = pd.DataFrame(columns={'sample_id', 'label'})
    result_predict_validation_all = pd.DataFrame(columns={'sample_id', 'label'})
    result_auc = pd.DataFrame(columns={'AUC_discovery', 'AUC_validation'})
    if len(set(label['y'])) > 2:
        for i in range(num):
            num_list = ['num_' + str(y_n) + '_' + str(i) for y_n in list(set(label['y']))]
            feature_type = 'count'
            result_predict_discovery = pd.read_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_discovery_' + feature_type + '_' + feature_name + '.txt',sep='\t')
            result_predict_validation = pd.read_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_validation_' + feature_type + '_' + feature_name + '.txt',sep='\t')
            result_predict_discovery = result_predict_discovery[['sample_id', 'label'] + num_list]
            result_predict_validation = result_predict_validation[['sample_id', 'label'] + num_list]
            for type in ['CNV', 'methylation']:
                print(type)
                result_predict_discovery_ = pd.read_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_discovery_' + type + '_' + feature_name + '.txt',sep='\t')
                result_predict_validation_ = pd.read_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_validation_' + type + '_' + feature_name + '.txt',sep='\t')
                for s in num_list:
                    result_predict_discovery[s] = result_predict_discovery[s] + result_predict_discovery_[s]
                    result_predict_validation[s] = result_predict_validation[s] + result_predict_validation_[s]

            y_discovery = np.array(result_predict_discovery['label'])
            y_discovery = y_discovery.astype('int')
            y_validation = np.array(result_predict_validation['label'])
            y_validation = y_validation.astype('int')

            predict_discovery = np.zeros([len(result_predict_discovery), len(set(label['y']))])
            predict_validation = np.zeros([len(result_predict_validation), len(set(label['y']))])

            for s in range(len(set(label['y']))):
                predict_discovery[:, s] = result_predict_discovery['num_' + str(s) + '_' + str(i)] / 7
                predict_validation[:, s] = result_predict_validation['num_' + str(s) + '_' + str(i)] / 7

            y_discovery_index = list(set(y_discovery))
            y_discovery_index.sort()
            y_validation_index = list(set(y_validation))
            y_validation_index.sort()

            predict_discovery_ = predict_discovery[:, y_discovery_index] / predict_discovery[:, y_discovery_index].sum(axis=1, keepdims=1)
            predict_validation_ = predict_validation[:, y_validation_index] / predict_validation[:,y_validation_index].sum(axis=1, keepdims=1)
            predict_discovery_[np.isnan(predict_discovery_)] = 1 / predict_discovery_.shape[1]
            predict_validation_[np.isnan(predict_validation_)] = 1 / predict_validation_.shape[1]
            acc_discovery, auc_weighted_ovr_discovery, auc_weighted_ovo_discovery, auc_macro_ovr_discovery, auc_macro_ovo_discovery, f1_weighted_discovery, f1_macro_discovery = plot_roc_multi( predict_discovery_, rankdata(y_discovery, method='dense') - 1)
            acc_validation, auc_weighted_ovr_validation, auc_weighted_ovo_validation, auc_macro_ovr_validation, auc_macro_ovo_validation, f1_weighted_validation, f1_macro_validation = plot_roc_multi(predict_validation_, rankdata(y_validation, method='dense') - 1)

            for y_num in list(set(y_discovery)):
                result_predict_discovery_all['num_' + str(y_num) + '_' + str(i)] = predict_discovery[:, y_num]
                result_predict_validation_all['num_' + str(y_num) + '_' + str(i)] = predict_validation[:, y_num]

            result_auc.loc[i, 'acc_discovery'] = acc_discovery
            result_auc.loc[i, 'auc_weighted_ovr_discovery'] = auc_weighted_ovr_discovery
            result_auc.loc[i, 'auc_weighted_ovo_discovery'] = auc_weighted_ovo_discovery
            result_auc.loc[i, 'auc_macro_ovr_discovery'] = auc_macro_ovr_discovery
            result_auc.loc[i, 'auc_macro_ovo_discovery'] = auc_macro_ovo_discovery
            result_auc.loc[i, 'f1_weighted_discovery'] = f1_weighted_discovery
            result_auc.loc[i, 'f1_macro_discovery'] = f1_macro_discovery
            result_auc.loc[i, 'acc_validation'] = acc_validation
            result_auc.loc[i, 'auc_weighted_ovr_validation'] = auc_weighted_ovr_validation
            result_auc.loc[i, 'auc_weighted_ovo_validation'] = auc_weighted_ovo_validation
            result_auc.loc[i, 'auc_macro_ovr_validation'] = auc_macro_ovr_validation
            result_auc.loc[i, 'auc_macro_ovo_validation'] = auc_macro_ovo_validation
            result_auc.loc[i, 'f1_weighted_validation'] = f1_weighted_validation
            result_auc.loc[i, 'f1_macro_validation'] = f1_macro_validation

        result_predict_discovery_all['sample_id'] = result_predict_discovery['sample_id']
        result_predict_validation_all['sample_id'] = result_predict_validation['sample_id']
        result_predict_discovery_all['label'] = result_predict_discovery['label']
        result_predict_validation_all['label'] = result_predict_validation['label']

    else:
        for i in range(num):
            num_list = ['num_' + str(i)]
            feature_type = 'count'
            result_predict_discovery = pd.read_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_discovery_' + feature_type + '_' + feature_name + '.txt',sep='\t')
            result_predict_validation = pd.read_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_validation_' + feature_type + '_' + feature_name + '.txt',sep='\t')
            result_predict_discovery = result_predict_discovery[['sample_id', 'label'] + num_list]
            result_predict_validation = result_predict_validation[['sample_id', 'label'] + num_list]
            for type in ['CNV', 'methylation']:
                print(type)
                result_predict_discovery_ = pd.read_csv(save_path + str(dataset) + '/' + str( method.split('_cv')[0]) + '/result_newnew_predict_discovery_' + type + '_' + feature_name + '.txt', sep='\t')
                result_predict_validation_ = pd.read_csv(save_path + str(dataset) + '/' + str( method.split('_cv')[0]) + '/result_newnew_predict_validation_' + type + '_' + feature_name + '.txt',sep='\t')
                for s in num_list:
                    result_predict_discovery[s] = result_predict_discovery[s] + result_predict_discovery_[s]
                    result_predict_validation[s] = result_predict_validation[s] + result_predict_validation_[s]
            y_discovery = np.array(result_predict_discovery['label'])
            y_discovery = y_discovery.astype('int')
            y_validation = np.array(result_predict_validation['label'])
            y_validation = y_validation.astype('int')

            predict_discovery = np.zeros([len(result_predict_discovery), 1])
            predict_validation = np.zeros([len(result_predict_validation), 1])

            predict_discovery[:, 0] = result_predict_discovery['num_' + str(i)] / 7
            predict_validation[:, 0] = result_predict_validation['num_' + str(i)] / 7

            acc_discovery, auc_discovery, f1_weighted_discovery, f1_macro_discovery = get_roc(y_discovery, predict_discovery[:, 0])
            acc_validation, auc_validation, f1_weighted_validation, f1_macro_validation = get_roc(y_validation, predict_validation[:, 0])

            result_predict_discovery_all['num_' + str(i)] = predict_discovery[:, 0]
            result_predict_validation_all['num_' + str(i)] = predict_validation[:, 0]
            result_auc.loc[i, 'acc_discovery'] = acc_discovery
            result_auc.loc[i, 'auc_discovery'] = auc_discovery
            result_auc.loc[i, 'f1_weighted_discovery'] = f1_weighted_discovery
            result_auc.loc[i, 'f1_macro_discovery'] = f1_macro_discovery
            result_auc.loc[i, 'acc_validation'] = acc_validation
            result_auc.loc[i, 'auc_validation'] = auc_validation
            result_auc.loc[i, 'f1_weighted_validation'] = f1_weighted_validation
            result_auc.loc[i, 'f1_macro_validation'] = f1_macro_validation

        result_predict_discovery_all['sample_id'] = result_predict_discovery['sample_id']
        result_predict_validation_all['sample_id'] = result_predict_validation['sample_id']
        result_predict_discovery_all['label'] = result_predict_discovery['label']
        result_predict_validation_all['label'] = result_predict_validation['label']

    result_predict_discovery.to_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_discovery_merge_all_' + feature_name + '.txt', sep='\t', index=False)
    result_predict_validation.to_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_validation_merge_all_' + feature_name + '.txt', sep='\t',index=False)
    result_auc.to_csv(save_path + str(dataset) + '/' + str(method.split('_cv')[0]) + '/result_newnew_AUC_merge_all_' + feature_name + '.txt',sep='\t', index=False)


def main(args):
    for type in ['count', 'CNV', 'methylation']:
        print(type)
        main_model(args.data_path, args.label_path, args.feature_name, args.dataset, args.method, args.num, args.save_path, type)
    merge(args.dataset, args.feature_name, args.method, args.num, args.save_path, args.label_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--data_path', type=str, required=True,
                        help='data_path', dest='data_path')
    parser.add_argument('--label_path', type=str, required=True,
                        help='label_path', dest='label_path')
    parser.add_argument('--feature_name', type=str, required=True,
                        help='feature_name', dest='feature_name')
    parser.add_argument('--dataset', type=int, required=True,
                        help='dataset', dest='dataset')
    parser.add_argument('--method', type=str, required=True,
                        help='method', dest='method')
    parser.add_argument('--num', type=int, required=True,
                        help='num', dest='num')
    parser.add_argument('--save_path', type=str, required=True,
                        help='save_path', dest='save_path')
    args = parser.parse_args()
    main(args)



