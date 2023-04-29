import pandas as pd
import numpy as np
import sklearn
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score, f1_score, \
    accuracy_score
import argparse, sys, os, errno

def plot_roc(prob, label):
    fpr, tpr, _ = roc_curve(label, prob)
    precision, recall, _ = precision_recall_curve(label, prob)  # recall: Identical to sensitivity,TPR; precision: PPV
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    # print (fpr, tpr,ppv)
    return roc_auc, fpr, tpr, precision, recall


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


def get_roc_multi(y, pro):
    y_index = list(set(y))
    y_index.sort()
    pro_ = pro[:, y_index] / pro[:, y_index].sum(axis=1, keepdims=1)
    pro_[np.isnan(pro_)] = 1 / pro_.shape[1]
    acc, auc_weighted_ovr, auc_weighted_ovo, auc_macro_ovr, auc_macro_ovo, f1_weighted, f1_macro = plot_roc_multi(
        pro_, rankdata(y, method='dense') - 1)
    return acc, auc_weighted_ovr, auc_weighted_ovo, auc_macro_ovr, auc_macro_ovo, f1_weighted, f1_macro


def get_roc(y, pro):
    label = np.array(y)
    pre_label = (pro > 0.5).astype(int)
    auc = roc_auc_score(label, np.array(pro))
    acc = accuracy_score(label, pre_label)
    f1_macro = f1_score(label, pre_label, average='macro')
    f1_weighted = f1_score(label, pre_label, average='weighted')
    return acc, auc, f1_weighted, f1_macro


def get_result(result_path):
    for c in ['BRCA_subtype', 'BRCA_stage', 'BRCA_survival']:
        print(c)
        for d in range(1, 11):
            print(d)
            for t in ['merge']:
                print(t)
                save_path = result_path+'/' + c + '/mixOmics/PLSDA/' + str(d) + '/' + t + '/'
                data = pd.read_csv(save_path + '/result_cv_2_new.csv', sep='\t', header=None)
                y_num = np.array(data.loc[0, data.columns[-1]].split(',')).astype('int').max() + 1
                if y_num == 2:
                    result = pd.DataFrame(columns={'roc_auc_Averaged_train', 'roc_auc_Averaged_test', 'roc_auc_Weighted_train','roc_auc_Weighted_test'})
                    data_result_all = pd.DataFrame(columns={'roc_auc_Averaged', 'roc_auc_Weighted'})
                    for m in [2, 5, 10]:
                        try:
                            data = pd.read_csv(save_path + '/result_cv_' + str(m) + '_new.csv', sep='\t', header=None)
                            data_result = pd.DataFrame(columns={'roc_auc_Averaged', 'roc_auc_Weighted'})
                            for i in range(0, 5):
                                data_cv = pd.DataFrame(data.iloc[i, :])
                                y_test = np.array(data_cv.iloc[y_num * 2 + 1][i].split(',')).astype('int')
                                predict_test_Averaged = np.zeros([len(y_test), y_num])
                                for j in range(y_num):
                                    predict_test_Averaged_ = np.array(data_cv.iloc[j][i].split(',')).astype('float')
                                    predict_test_Averaged[:, j] = predict_test_Averaged_

                                predict_test_Weighted = np.zeros([len(y_test), y_num])
                                for j in range(y_num):
                                    predict_test_Weighted_ = np.array(data_cv.iloc[j + y_num][i].split(',')).astype('float')
                                    predict_test_Weighted[:, j] = predict_test_Weighted_
                                predict_test_Averaged[np.isnan(predict_test_Averaged)] = 0.5
                                predict_test_Weighted[np.isnan(predict_test_Weighted)] = 0.5
                                roc_auc_Averaged, fpr_Averaged, tpr_Averaged, precision_Averaged, recall_Averaged = plot_roc(predict_test_Averaged[:, 1], y_test)
                                roc_auc_Weighted, fpr_Weighted, tpr_Weighted, precision_Weighted, recall_Weighted = plot_roc(predict_test_Weighted[:, 1], y_test)
                                data_result.loc[i, 'roc_auc_Averaged'] = roc_auc_Averaged
                                data_result.loc[i, 'roc_auc_Weighted'] = roc_auc_Weighted
                            data_result['num'] = m
                            data_result_all = pd.concat([data_result_all, data_result])
                        except FileNotFoundError:
                            continue

                    data_result_all = data_result_all.astype('float')
                    data_result_mean = data_result_all.groupby('num').mean()
                    max_num = int(list(data_result_mean.sort_values('roc_auc_Averaged').index)[-1])

                    data_validation = pd.read_csv(save_path + '/result_validation_' + str(max_num) + '_new.csv', sep='\t',header=None)
                    y_train = np.array(list(data_validation[0])[-2].split(',')).astype('int')
                    y_test = np.array(list(data_validation[0])[-1].split(',')).astype('int')

                    predict_train_Averaged = np.zeros([len(y_train), y_num])
                    predict_test_Averaged = np.zeros([len(y_test), y_num])
                    predict_train_Weighted = np.zeros([len(y_train), y_num])
                    predict_test_Weighted = np.zeros([len(y_test), y_num])
                    for j in range(y_num):
                        predict_train_Averaged_ = np.array(list(data_validation.iloc[j])[0].split(',')).astype('float')
                        predict_train_Averaged[:, j] = predict_train_Averaged_
                    for j in range(y_num):
                        predict_test_Averaged_ = np.array(list(data_validation.iloc[j + y_num])[0].split(',')).astype('float')
                        predict_test_Averaged[:, j] = predict_test_Averaged_
                    for j in range(y_num):
                        predict_train_Weighted_ = np.array(list(data_validation.iloc[j + y_num + y_num])[0].split(',')).astype('float')
                        predict_train_Weighted[:, j] = predict_train_Weighted_
                    for j in range(y_num):
                        predict_test_Weighted_ = np.array(list(data_validation.iloc[j + y_num + y_num + y_num])[0].split(',')).astype('float')
                        predict_test_Weighted[:, j] = predict_test_Weighted_
                    predict_train_Averaged[np.isnan(predict_train_Averaged)] = 0.5
                    predict_test_Averaged[np.isnan(predict_test_Averaged)] = 0.5
                    predict_train_Weighted[np.isnan(predict_train_Weighted)] = 0.5
                    predict_test_Weighted[np.isnan(predict_test_Weighted)] = 0.5
                    acc_Averaged_train, auc_Averaged_train, f1_weighted_Averaged_train, f1_macro_Averaged_train = get_roc(np.array(y_train), predict_train_Averaged[:, 1])
                    acc_Averaged_test, auc_Averaged_test, f1_weighted_Averaged_test, f1_macro_Averaged_test = get_roc(np.array(y_test), predict_test_Averaged[:, 1])
                    acc_Weighted_train, auc_Weighted_train, f1_weighted_Weighted_train, f1_macro_Weighted_train = get_roc(np.array(y_train), predict_train_Weighted[:, 1])
                    acc_Weighted_test, auc_Weighted_test, f1_weighted_Weighted_test, f1_macro_Weighted_test = get_roc(np.array(y_test), predict_test_Weighted[:, 1])
                    result.loc[0, 'acc_Averaged_train'] = acc_Averaged_train
                    result.loc[0, 'auc_Averaged_train'] = auc_Averaged_train
                    result.loc[0, 'f1_weighted_Averaged_train'] = f1_weighted_Averaged_train
                    result.loc[0, 'f1_macro_Averaged_train'] = f1_macro_Averaged_train
                    result.loc[0, 'acc_Averaged_test'] = acc_Averaged_test
                    result.loc[0, 'auc_Averaged_test'] = auc_Averaged_test
                    result.loc[0, 'f1_weighted_Averaged_test'] = f1_weighted_Averaged_test
                    result.loc[0, 'f1_macro_Averaged_test'] = f1_macro_Averaged_test
                    result.loc[0, 'acc_Weighted_train'] = acc_Weighted_train
                    result.loc[0, 'auc_Weighted_train'] = auc_Weighted_train
                    result.loc[0, 'f1_weighted_Weighted_train'] = f1_weighted_Weighted_train
                    result.loc[0, 'f1_macro_Weighted_train'] = f1_macro_Weighted_train
                    result.loc[0, 'acc_Weighted_test'] = acc_Weighted_test
                    result.loc[0, 'auc_Weighted_test'] = auc_Weighted_test
                    result.loc[0, 'f1_weighted_Weighted_test'] = f1_weighted_Weighted_test
                    result.loc[0, 'f1_macro_Weighted_test'] = f1_macro_Weighted_test

                else:
                    result = pd.DataFrame(columns={'auc_weighted_ovr_Averaged_train'})
                    data_result_all = pd.DataFrame(columns={'auc_macro_ovr_Averaged', 'auc_macro_ovr_Weighted', 'f1_macro_Averaged','f1_macro_Weighted'})
                    for m in [2, 5, 10]:
                        try:
                            data = pd.read_csv(save_path + '/result_cv_' + str(m) + '_new.csv', sep='\t', header=None)
                            data_result = pd.DataFrame(columns={'auc_macro_ovr_Averaged', 'auc_macro_ovr_Weighted', 'f1_macro_Averaged','f1_macro_Weighted'})
                            for i in range(0, 5):
                                data_cv = pd.DataFrame(data.iloc[i, :])
                                y_test = np.array(data_cv.iloc[y_num * 2 + 1][i].split(',')).astype('int')
                                predict_test_Averaged = np.zeros([len(y_test), y_num])
                                for j in range(y_num):
                                    predict_test_Averaged_ = np.array(data_cv.iloc[j][i].split(',')).astype('float')
                                    predict_test_Averaged[:, j] = predict_test_Averaged_
                                predict_test_Weighted = np.zeros([len(y_test), y_num])
                                for j in range(y_num):
                                    predict_test_Weighted_ = np.array(data_cv.iloc[j + y_num][i].split(',')).astype('float')
                                    predict_test_Weighted[:, j] = predict_test_Weighted_
                                predict_test_Averaged[np.isnan(predict_test_Averaged)] = 0.5
                                predict_test_Weighted[np.isnan(predict_test_Weighted)] = 0.5
                                y_test_index = list(set(y_test))
                                y_test_index.sort()
                                predict_test_Averaged_new = predict_test_Averaged[:, y_test_index] / predict_test_Averaged[:, y_test_index].sum(axis=1, keepdims=1)
                                predict_test_Weighted_new = predict_test_Weighted[:, y_test_index] / predict_test_Weighted[:, y_test_index].sum(axis=1, keepdims=1)
                                predict_test_Averaged_new[np.isnan(predict_test_Averaged_new)] = 1 / predict_test_Averaged_new.shape[1]
                                predict_test_Weighted_new[np.isnan(predict_test_Weighted_new)] = 1 / predict_test_Weighted_new.shape[1]
                                acc_Averaged, auc_weighted_ovr_Averaged, auc_weighted_ovo_Averaged, auc_macro_ovr_Averaged, auc_macro_ovo_Averaged, f1_weighted_Averaged, f1_macro_Averaged = plot_roc_multi(predict_test_Averaged_new, rankdata(y_test, method='dense') - 1)
                                acc_Weighted, auc_weighted_ovr_Weighted, auc_weighted_ovo_Weighted, auc_macro_ovr_Weighted, auc_macro_ovo_Weighted, f1_weighted_Weighted, f1_macro_Weighted = plot_roc_multi(predict_test_Weighted_new, rankdata(y_test, method='dense') - 1)
                                data_result.loc[i, 'auc_macro_ovr_Averaged'] = auc_macro_ovr_Averaged
                                data_result.loc[i, 'auc_macro_ovr_Weighted'] = auc_macro_ovr_Weighted
                                data_result.loc[i, 'f1_macro_Averaged'] = f1_macro_Averaged
                                data_result.loc[i, 'f1_macro_Weighted'] = f1_macro_Weighted
                            data_result['num'] = m
                            data_result_all = pd.concat([data_result_all, data_result])
                        except FileNotFoundError:
                            continue

                    data_result_all = data_result_all.astype('float')
                    data_result_mean = data_result_all.groupby('num').mean()
                    max_num = int(list(data_result_mean.sort_values('f1_macro_Averaged').index)[-1])

                    data_validation = pd.read_csv(save_path + '/result_validation_' + str(max_num) + '_new.csv', sep='\t',header=None)
                    y_train = np.array(list(data_validation[0])[-2].split(',')).astype('int')
                    y_test = np.array(list(data_validation[0])[-1].split(',')).astype('int')

                    predict_train_Averaged = np.zeros([len(y_train), y_num])
                    predict_test_Averaged = np.zeros([len(y_test), y_num])
                    predict_train_Weighted = np.zeros([len(y_train), y_num])
                    predict_test_Weighted = np.zeros([len(y_test), y_num])
                    for j in range(y_num):
                        predict_train_Averaged_ = np.array(list(data_validation.iloc[j])[0].split(',')).astype('float')
                        predict_train_Averaged[:, j] = predict_train_Averaged_
                    for j in range(y_num):
                        predict_test_Averaged_ = np.array(list(data_validation.iloc[j + y_num])[0].split(',')).astype('float')
                        predict_test_Averaged[:, j] = predict_test_Averaged_
                    for j in range(y_num):
                        predict_train_Weighted_ = np.array(list(data_validation.iloc[j + y_num + y_num])[0].split(',')).astype('float')
                        predict_train_Weighted[:, j] = predict_train_Weighted_
                    for j in range(y_num):
                        predict_test_Weighted_ = np.array(list(data_validation.iloc[j + y_num + y_num + y_num])[0].split(',')).astype('float')
                        predict_test_Weighted[:, j] = predict_test_Weighted_

                    predict_train_Averaged[np.isnan(predict_train_Averaged)] = 0.5
                    predict_test_Averaged[np.isnan(predict_test_Averaged)] = 0.5
                    predict_train_Weighted[np.isnan(predict_train_Weighted)] = 0.5
                    predict_test_Weighted[np.isnan(predict_test_Weighted)] = 0.5
                    y_test_index = list(set(y_test))
                    y_test_index.sort()
                    predict_test_Averaged_new = predict_test_Averaged[:, y_test_index] / predict_test_Averaged[:,y_test_index].sum(axis=1,keepdims=1)
                    predict_test_Weighted_new = predict_test_Weighted[:, y_test_index] / predict_test_Weighted[:,y_test_index].sum(axis=1,keepdims=1)
                    predict_test_Averaged_new[np.isnan(predict_test_Averaged_new)] = 1 / predict_test_Averaged_new.shape[1]
                    predict_test_Weighted_new[np.isnan(predict_test_Weighted_new)] = 1 / predict_test_Weighted_new.shape[1]
                    acc_Averaged_test, auc_weighted_ovr_Averaged_test, auc_weighted_ovo_Averaged_test, auc_macro_ovr_Averaged_test, auc_macro_ovo_Averaged_test, f1_weighted_Averaged_test, f1_macro_Averaged_test = plot_roc_multi(predict_test_Averaged_new, rankdata(y_test, method='dense') - 1)
                    acc_Weighted_test, auc_weighted_ovr_Weighted_test, auc_weighted_ovo_Weighted_test, auc_macro_ovr_Weighted_test, auc_macro_ovo_Weighted_test, f1_weighted_Weighted_test, f1_macro_Weighted_test = plot_roc_multi(predict_test_Weighted_new, rankdata(y_test, method='dense') - 1)
                    y_train_index = list(set(y_train))
                    y_train_index.sort()
                    predict_train_Averaged_new = predict_train_Averaged[:, y_train_index] / predict_train_Averaged[:,y_train_index].sum(axis=1,keepdims=1)
                    predict_train_Weighted_new = predict_train_Weighted[:, y_train_index] / predict_train_Weighted[:,y_train_index].sum(axis=1,keepdims=1)
                    predict_train_Averaged_new[np.isnan(predict_train_Averaged_new)] = 1 / predict_train_Averaged_new.shape[1]
                    predict_train_Weighted_new[np.isnan(predict_train_Weighted_new)] = 1 / predict_train_Weighted_new.shape[1]
                    acc_Averaged_train, auc_weighted_ovr_Averaged_train, auc_weighted_ovo_Averaged_train, auc_macro_ovr_Averaged_train, auc_macro_ovo_Averaged_train, f1_weighted_Averaged_train, f1_macro_Averaged_train = plot_roc_multi(
                        predict_train_Averaged_new, rankdata(y_train, method='dense') - 1)
                    acc_Weighted_train, auc_weighted_ovr_Weighted_train, auc_weighted_ovo_Weighted_train, auc_macro_ovr_Weighted_train, auc_macro_ovo_Weighted_train, f1_weighted_Weighted_train, f1_macro_Weighted_train = plot_roc_multi(
                        predict_train_Weighted_new, rankdata(y_train, method='dense') - 1)

                    result.loc[0, 'acc_Averaged_train'] = acc_Averaged_train
                    result.loc[0, 'auc_weighted_ovr_Averaged_train'] = auc_weighted_ovr_Averaged_train
                    result.loc[0, 'auc_weighted_ovo_Averaged_train'] = auc_weighted_ovo_Averaged_train
                    result.loc[0, 'auc_macro_ovr_Averaged_train'] = auc_macro_ovr_Averaged_train
                    result.loc[0, 'auc_macro_ovo_Averaged_train'] = auc_macro_ovo_Averaged_train
                    result.loc[0, 'f1_weighted_Averaged_train'] = f1_weighted_Averaged_train
                    result.loc[0, 'f1_macro_Averaged_train'] = f1_macro_Averaged_train
                    result.loc[0, 'acc_Averaged_test'] = acc_Averaged_test
                    result.loc[0, 'auc_weighted_ovr_Averaged_test'] = auc_weighted_ovr_Averaged_test
                    result.loc[0, 'auc_weighted_ovo_Averaged_test'] = auc_weighted_ovo_Averaged_test
                    result.loc[0, 'auc_macro_ovr_Averaged_test'] = auc_macro_ovr_Averaged_test
                    result.loc[0, 'auc_macro_ovo_Averaged_test'] = auc_macro_ovo_Averaged_test
                    result.loc[0, 'f1_weighted_Averaged_test'] = f1_weighted_Averaged_test
                    result.loc[0, 'f1_macro_Averaged_test'] = f1_macro_Averaged_test
                    result.loc[0, 'acc_Weighted_train'] = acc_Weighted_train
                    result.loc[0, 'auc_weighted_ovr_Weighted_train'] = auc_weighted_ovr_Weighted_train
                    result.loc[0, 'auc_weighted_ovo_Weighted_train'] = auc_weighted_ovo_Weighted_train
                    result.loc[0, 'auc_macro_ovr_Weighted_train'] = auc_macro_ovr_Weighted_train
                    result.loc[0, 'auc_macro_ovo_Weighted_train'] = auc_macro_ovo_Weighted_train
                    result.loc[0, 'f1_weighted_Weighted_train'] = f1_weighted_Weighted_train
                    result.loc[0, 'f1_macro_Weighted_train'] = f1_macro_Weighted_train
                    result.loc[0, 'acc_Weighted_test'] = acc_Weighted_test
                    result.loc[0, 'auc_weighted_ovr_Weighted_test'] = auc_weighted_ovr_Weighted_test
                    result.loc[0, 'auc_weighted_ovo_Weighted_test'] = auc_weighted_ovo_Weighted_test
                    result.loc[0, 'auc_macro_ovr_Weighted_test'] = auc_macro_ovr_Weighted_test
                    result.loc[0, 'auc_macro_ovo_Weighted_test'] = auc_macro_ovo_Weighted_test
                    result.loc[0, 'f1_weighted_Weighted_test'] = f1_weighted_Weighted_test
                    result.loc[0, 'f1_macro_Weighted_test'] = f1_macro_Weighted_test
                result.to_csv(save_path + 'result_AUC_new.csv', sep='\t', index=False)

def main(args):
    get_result(args.result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data module')
    parser.add_argument('--result_path', type=str, required=True,
                        help='result_path', dest='result_path')
    args = parser.parse_args()
    main(args)