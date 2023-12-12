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
    sample_train = label.loc[label['dataset_' + str(dataset)] != 'test', 'sample_id']
    sample_test = label.loc[label['dataset_' + str(dataset)] == 'test', 'sample_id']
    label = label.set_index('sample_id')
    y_train = np.array(label.loc[sample_train, 'y'])
    y_train = y_train.astype('int')
    y_test = np.array(label.loc[sample_test, 'y'])
    y_test = y_test.astype('int')
    X_train = np.array(data.loc[:, sample_train]).T
    X_test = np.array(data.loc[:, sample_test]).T
    result_predict_train = pd.DataFrame(columns={'sample_id', 'label'})
    result_predict_train['sample_id'] = sample_train
    result_predict_train['label'] = list(label.loc[sample_train, 'y'])
    result_predict_test = pd.DataFrame(columns={'sample_id', 'label'})
    result_predict_test['sample_id'] = sample_test
    result_predict_test['label'] = list(label.loc[sample_test, 'y'])
    result_auc = pd.DataFrame(columns={'AUC_train', 'AUC_test'})
    pds = 5
    for i in range(num):
        sample_weight_ = compute_sample_weight(class_weight='balanced', y=y_train)
        if len(set(label['y'])) > 2:
            clf = clf_select_multi(method, pds)
        else:
            clf = clf_select(method, pds)
        if (method == 'KNN') | (method == 'KNN_cv') | (method == 'NN'):
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train, sample_weight_)

        if len(set(label['y'])) > 2:
            predict_train = clf.predict_proba(X_train)
            predict_test = clf.predict_proba(X_test)
            y_train_index = list(set(y_train))
            y_train_index.sort()
            y_test_index = list(set(y_test))
            y_test_index.sort()
            predict_train_ = predict_train[:, y_train_index] / predict_train[:, y_train_index].sum(axis=1, keepdims=1)
            predict_test_ = predict_test[:, y_test_index] / predict_test[:,y_test_index].sum(axis=1,keepdims=1)
            predict_train_[np.isnan(predict_train_)] = 1 / predict_train_.shape[1]
            predict_test_[np.isnan(predict_test_)] = 1 / predict_test_.shape[1]
            acc_train, auc_weighted_ovr_train, auc_weighted_ovo_train, auc_macro_ovr_train, auc_macro_ovo_train, f1_weighted_train, f1_macro_train = plot_roc_multi(
                predict_train_, rankdata(y_train, method='dense') - 1)
            acc_test, auc_weighted_ovr_test, auc_weighted_ovo_test, auc_macro_ovr_test, auc_macro_ovo_test, f1_weighted_test, f1_macro_test = plot_roc_multi(
                predict_test_, rankdata(y_test, method='dense') - 1)
            for y_num in list(set(y_train)):
                result_predict_train['num_' + str(y_num) + '_' + str(i)] = predict_train[:, y_num]
                result_predict_test['num_' + str(y_num) + '_' + str(i)] = predict_test[:, y_num]
            result_auc.loc[i, 'acc_train'] = acc_train
            result_auc.loc[i, 'auc_weighted_ovr_train'] = auc_weighted_ovr_train
            result_auc.loc[i, 'auc_weighted_ovo_train'] = auc_weighted_ovo_train
            result_auc.loc[i, 'auc_macro_ovr_train'] = auc_macro_ovr_train
            result_auc.loc[i, 'auc_macro_ovo_train'] = auc_macro_ovo_train
            result_auc.loc[i, 'f1_weighted_train'] = f1_weighted_train
            result_auc.loc[i, 'f1_macro_train'] = f1_macro_train
            result_auc.loc[i, 'acc_test'] = acc_test
            result_auc.loc[i, 'auc_weighted_ovr_test'] = auc_weighted_ovr_test
            result_auc.loc[i, 'auc_weighted_ovo_test'] = auc_weighted_ovo_test
            result_auc.loc[i, 'auc_macro_ovr_test'] = auc_macro_ovr_test
            result_auc.loc[i, 'auc_macro_ovo_test'] = auc_macro_ovo_test
            result_auc.loc[i, 'f1_weighted_test'] = f1_weighted_test
            result_auc.loc[i, 'f1_macro_test'] = f1_macro_test
        else:
            predict_train = clf.predict_proba(X_train)
            roc_auc_train, fpr_train, tpr_train, precision_train, recall_train = plot_roc(predict_train[:, 1], y_train)
            predict_test = clf.predict_proba(X_test)
            roc_auc_test, fpr_test, tpr_test, precision_test, recall_test = plot_roc(predict_test[:, 1], y_test)
            result_predict_train['num_' + str(i)] = predict_train[:, 1]
            result_predict_test['num_' + str(i)] = predict_test[:, 1]
            result_auc.loc[i, 'AUC_train'] = roc_auc_train
            result_auc.loc[i, 'AUC_test'] = roc_auc_test
    result_predict_train.to_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_train_' + type + '_' + feature_name + '.txt', sep='\t',index=False)
    result_predict_test.to_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_test_' + type + '_' + feature_name + '.txt', sep='\t',index=False)
    result_auc.to_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_AUC_' + type + '_' + feature_name + '.txt', sep='\t', index=False)


def merge(dataset, feature_name, method, num, save_path, label_path):
    label = pd.read_csv(label_path, sep='\t')
    result_predict_train_all = pd.DataFrame(columns={'sample_id', 'label'})
    result_predict_test_all = pd.DataFrame(columns={'sample_id', 'label'})
    result_auc = pd.DataFrame(columns={'AUC_train', 'AUC_test'})
    if len(set(label['y'])) > 2:
        for i in range(num):
            num_list = ['num_' + str(y_n) + '_' + str(i) for y_n in list(set(label['y']))]
            feature_type = 'count'
            result_predict_train = pd.read_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_train_' + feature_type + '_' + feature_name + '.txt',sep='\t')
            result_predict_test = pd.read_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_test_' + feature_type + '_' + feature_name + '.txt',sep='\t')
            result_predict_train = result_predict_train[['sample_id', 'label'] + num_list]
            result_predict_test = result_predict_test[['sample_id', 'label'] + num_list]
            for type in ['CNV', 'methylation']:
                print(type)
                result_predict_train_ = pd.read_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_train_' + type + '_' + feature_name + '.txt',sep='\t')
                result_predict_test_ = pd.read_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_test_' + type + '_' + feature_name + '.txt',sep='\t')
                for s in num_list:
                    result_predict_train[s] = result_predict_train[s] + result_predict_train_[s]
                    result_predict_test[s] = result_predict_test[s] + result_predict_test_[s]

            y_train = np.array(result_predict_train['label'])
            y_train = y_train.astype('int')
            y_test = np.array(result_predict_test['label'])
            y_test = y_test.astype('int')

            predict_train = np.zeros([len(result_predict_train), len(set(label['y']))])
            predict_test = np.zeros([len(result_predict_test), len(set(label['y']))])

            for s in range(len(set(label['y']))):
                predict_train[:, s] = result_predict_train['num_' + str(s) + '_' + str(i)] / 7
                predict_test[:, s] = result_predict_test['num_' + str(s) + '_' + str(i)] / 7

            y_train_index = list(set(y_train))
            y_train_index.sort()
            y_test_index = list(set(y_test))
            y_test_index.sort()

            predict_train_ = predict_train[:, y_train_index] / predict_train[:, y_train_index].sum(axis=1, keepdims=1)
            predict_test_ = predict_test[:, y_test_index] / predict_test[:,y_test_index].sum(axis=1, keepdims=1)
            predict_train_[np.isnan(predict_train_)] = 1 / predict_train_.shape[1]
            predict_test_[np.isnan(predict_test_)] = 1 / predict_test_.shape[1]
            acc_train, auc_weighted_ovr_train, auc_weighted_ovo_train, auc_macro_ovr_train, auc_macro_ovo_train, f1_weighted_train, f1_macro_train = plot_roc_multi( predict_train_, rankdata(y_train, method='dense') - 1)
            acc_test, auc_weighted_ovr_test, auc_weighted_ovo_test, auc_macro_ovr_test, auc_macro_ovo_test, f1_weighted_test, f1_macro_test = plot_roc_multi(predict_test_, rankdata(y_test, method='dense') - 1)

            for y_num in list(set(y_train)):
                result_predict_train_all['num_' + str(y_num) + '_' + str(i)] = predict_train[:, y_num]
                result_predict_test_all['num_' + str(y_num) + '_' + str(i)] = predict_test[:, y_num]

            result_auc.loc[i, 'acc_train'] = acc_train
            result_auc.loc[i, 'auc_weighted_ovr_train'] = auc_weighted_ovr_train
            result_auc.loc[i, 'auc_weighted_ovo_train'] = auc_weighted_ovo_train
            result_auc.loc[i, 'auc_macro_ovr_train'] = auc_macro_ovr_train
            result_auc.loc[i, 'auc_macro_ovo_train'] = auc_macro_ovo_train
            result_auc.loc[i, 'f1_weighted_train'] = f1_weighted_train
            result_auc.loc[i, 'f1_macro_train'] = f1_macro_train
            result_auc.loc[i, 'acc_test'] = acc_test
            result_auc.loc[i, 'auc_weighted_ovr_test'] = auc_weighted_ovr_test
            result_auc.loc[i, 'auc_weighted_ovo_test'] = auc_weighted_ovo_test
            result_auc.loc[i, 'auc_macro_ovr_test'] = auc_macro_ovr_test
            result_auc.loc[i, 'auc_macro_ovo_test'] = auc_macro_ovo_test
            result_auc.loc[i, 'f1_weighted_test'] = f1_weighted_test
            result_auc.loc[i, 'f1_macro_test'] = f1_macro_test

        result_predict_train_all['sample_id'] = result_predict_train['sample_id']
        result_predict_test_all['sample_id'] = result_predict_test['sample_id']
        result_predict_train_all['label'] = result_predict_train['label']
        result_predict_test_all['label'] = result_predict_test['label']

    else:
        for i in range(num):
            num_list = ['num_' + str(i)]
            feature_type = 'count'
            result_predict_train = pd.read_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_train_' + feature_type + '_' + feature_name + '.txt',sep='\t')
            result_predict_test = pd.read_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_test_' + feature_type + '_' + feature_name + '.txt',sep='\t')
            result_predict_train = result_predict_train[['sample_id', 'label'] + num_list]
            result_predict_test = result_predict_test[['sample_id', 'label'] + num_list]
            for type in ['CNV', 'methylation']:
                print(type)
                result_predict_train_ = pd.read_csv(save_path + '/' + str( method.split('_cv')[0]) + '/result_newnew_predict_train_' + type + '_' + feature_name + '.txt', sep='\t')
                result_predict_test_ = pd.read_csv(save_path + '/' + str( method.split('_cv')[0]) + '/result_newnew_predict_test_' + type + '_' + feature_name + '.txt',sep='\t')
                for s in num_list:
                    result_predict_train[s] = result_predict_train[s] + result_predict_train_[s]
                    result_predict_test[s] = result_predict_test[s] + result_predict_test_[s]
            y_train = np.array(result_predict_train['label'])
            y_train = y_train.astype('int')
            y_test = np.array(result_predict_test['label'])
            y_test = y_test.astype('int')

            predict_train = np.zeros([len(result_predict_train), 1])
            predict_test = np.zeros([len(result_predict_test), 1])

            predict_train[:, 0] = result_predict_train['num_' + str(i)] / 7
            predict_test[:, 0] = result_predict_test['num_' + str(i)] / 7

            acc_train, auc_train, f1_weighted_train, f1_macro_train = get_roc(y_train, predict_train[:, 0])
            acc_test, auc_test, f1_weighted_test, f1_macro_test = get_roc(y_test, predict_test[:, 0])

            result_predict_train_all['num_' + str(i)] = predict_train[:, 0]
            result_predict_test_all['num_' + str(i)] = predict_test[:, 0]
            result_auc.loc[i, 'acc_train'] = acc_train
            result_auc.loc[i, 'auc_train'] = auc_train
            result_auc.loc[i, 'f1_weighted_train'] = f1_weighted_train
            result_auc.loc[i, 'f1_macro_train'] = f1_macro_train
            result_auc.loc[i, 'acc_test'] = acc_test
            result_auc.loc[i, 'auc_test'] = auc_test
            result_auc.loc[i, 'f1_weighted_test'] = f1_weighted_test
            result_auc.loc[i, 'f1_macro_test'] = f1_macro_test

        result_predict_train_all['sample_id'] = result_predict_train['sample_id']
        result_predict_test_all['sample_id'] = result_predict_test['sample_id']
        result_predict_train_all['label'] = result_predict_train['label']
        result_predict_test_all['label'] = result_predict_test['label']

    result_predict_train.to_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_train_merge_all_' + feature_name + '.txt', sep='\t', index=False)
    result_predict_test.to_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_predict_test_merge_all_' + feature_name + '.txt', sep='\t',index=False)
    result_auc.to_csv(save_path + '/' + str(method.split('_cv')[0]) + '/result_newnew_AUC_merge_all_' + feature_name + '.txt',sep='\t', index=False)


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
    
