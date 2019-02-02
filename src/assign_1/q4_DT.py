import sys
import os
import pprint
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import KFold as KF
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
from IPython.display import Image
import pydotplus


pd.set_option('display.expand_frame_repr', False)


def load_data(trainPath, testPath):

    train_df = pd.read_csv(trainPath, delimiter=",", header=None, index_col=0)
    test_df = pd.read_csv(testPath, delimiter=",", header=None, index_col=0)

    # print(len(test_df.groupby([1]).groups['M']))
    # B = 250, M = 149 --> train
    # B = 63, M = 107 --> test
    return train_df, test_df


def DT_model(trainPath, testPath, max_depth_range, criteria_dt):
    train_df, test_df = load_data(trainPath, testPath)
    DT_score_cv = {}

    for each in range(1, max_depth_range + 1):
        dtc = DTC(criterion=criteria_dt, max_depth=each)
        cv_kf = KF(n_splits=10, shuffle=True, random_state=123)
        split_cv = cv_kf.split(train_df)
        cross_val_scores = {}
        for i, eachSplit in enumerate(split_cv):
            train_cv = train_df.iloc[eachSplit[0]]
            test_cv = train_df.iloc[eachSplit[1]]
            dt_cv = dtc.fit(train_cv[train_cv.columns.drop(
                [1])], train_cv[1])
            y_pred = dt_cv.predict(test_cv[test_cv.columns.drop(
                [1])])
            tn, fp, fn, tp = confusion_matrix(test_cv[1], y_pred).ravel()
            recall_cv = tp / (tp + fn)
            sensitivity_cv = recall_cv
            precision_cv = tp / (tp + fp)
            specificity_cv = tn / (fp + tn)

            cross_val_scores.setdefault("recall_cv", []).append(recall_cv)
            cross_val_scores.setdefault(
                "precision_cv", []).append(precision_cv)
            cross_val_scores.setdefault(
                "sensitivity_cv", []).append(sensitivity_cv)
            cross_val_scores.setdefault(
                "specificity_cv", []).append(specificity_cv)

        recall_avg = round(((sum(
            cross_val_scores["recall_cv"]) / len(cross_val_scores['recall_cv'])) * 100), 2)
        precision_avg = round(((sum(
            cross_val_scores["precision_cv"]) / len(cross_val_scores['precision_cv'])) * 100), 2)
        sensitivity_avg = round(((sum(
            cross_val_scores["sensitivity_cv"]) / len(cross_val_scores['sensitivity_cv'])) * 100), 2)
        specificity_avg = round(((sum(
            cross_val_scores["specificity_cv"]) / len(cross_val_scores['specificity_cv'])) * 100), 2)

        DT_score_cv[each] = {"recall_avg_cv": recall_avg, 'precision_avg_cv': precision_avg,
                             'sensitivity_avg_cv': sensitivity_avg, 'specificity_avg_cv': specificity_avg}

    DT_train_test_scores = {}
    for each in range(1, max_depth_range + 1):
        dtc = DTC(criterion=criteria_dt, max_depth=each)
        train2_df, test2_df = load_data(trainPath, testPath)
        DT_tt = dtc.fit(train2_df[train_df.columns.drop(
            [1])], train2_df[1])
        y_pred2 = DT_tt.predict(test2_df[test2_df.columns.drop(
            [1])])
        tn2, fp2, fn2, tp2 = confusion_matrix(test2_df[1], y_pred2).ravel()
        recall_tt = round(((tp2 / (tp2 + fn2)) * 100), 2)
        sensitivity_tt = recall_tt
        precision_tt = round(((tp2 / (tp2 + fp2)) * 100), 2)
        specificity_tt = round(((tn2 / (fp2 + tn2)) * 100), 2)

        DT_train_test_scores[each] = {"recall_model": recall_tt, 'precision_model': precision_tt,
                                      'sensitivity_model': sensitivity_tt, 'specificity_model': specificity_tt}

    return DT_score_cv, DT_train_test_scores


def plot_score(trainPath, testPath, max_depth_range, criteria_dt):
    dt_score_cv, dt_train_test_scores = DT_model(
        trainPath, testPath, max_depth_range, criteria_dt)
    recall_cvavg_lst = []
    precision_cvavg_lst = []
    specificity_cvavg_lst = []
    for k, v in dt_score_cv.items():
        recall_cvavg_lst.append(v['recall_avg_cv'])
        precision_cvavg_lst.append(v['precision_avg_cv'])
        specificity_cvavg_lst.append(v['specificity_avg_cv'])

    cv_plt = plt.figure()
    plt.plot(dt_score_cv.keys(), recall_cvavg_lst)
    plt.plot(dt_score_cv.keys(), precision_cvavg_lst)
    plt.plot(dt_score_cv.keys(), specificity_cvavg_lst)
    # plt.plot(knn_score_cv.keys(), specificity_model_lst)
    cv_plt.suptitle(
        'DT-10 10-Fold CrossVal Recall, Precision, Specificity', fontsize=12)
    cv_plt.legend(['Recall_CrossVal', 'Precision_CrossVal',
                   'Specificity_CrossVal'], loc='lower left')
    plt.ylim(50, 100)
    plt.ylabel('Percent (%)')
    plt.xlabel('Depth of Tree')
    plt.show('hold')

    recall_model_lst = []
    precision_model_lst = []
    specificity_model_lst = []

    for k, v in dt_train_test_scores.items():
        recall_model_lst.append(v['recall_model'])
        precision_model_lst.append(v['precision_model'])
        specificity_model_lst.append(v['specificity_model'])

    model_plt = plt.figure()
    plt.plot(dt_score_cv.keys(), recall_model_lst)
    plt.plot(dt_score_cv.keys(), precision_model_lst)
    plt.plot(dt_score_cv.keys(), specificity_model_lst)
    # plt.plot(knn_score_cv.keys(), specificity_model_lst)
    model_plt.suptitle(
        'DT-10 10-Fold Model Recall, Precision, Specificity', fontsize=12)
    model_plt.legend(['Recall_Model', 'Precision_Model',
                      'Specificity_Model'], loc='lower left')
    plt.ylim(50, 100)
    plt.ylabel('Percent (%)')
    plt.xlabel('Depth of Tree')
    plt.show('hold')

    cv_model_spcf_plt = plt.figure()
    plt.plot(dt_score_cv.keys(), specificity_cvavg_lst)
    plt.plot(dt_score_cv.keys(), specificity_model_lst)
    cv_model_spcf_plt.suptitle(
        'DT-10 10-Fold CrossVal Vs Model Specificity', fontsize=12)
    # model_plt.legend(['Specificity_Model'], loc='lower left')
    plt.ylim(50, 100)
    plt.ylabel('Specificity (%)')
    plt.xlabel('Depth of Tree')
    plt.show('hold')


def final_model(trainPath, testPath, max_depth_final):
    train_df, test_df = load_data(trainPath, testPath)
    dtc = DTC(criterion='gini', max_depth=max_depth_final)
    dt_final = dtc.fit(train_df[train_df.columns.drop(
        [1])], train_df[1])
    dot_data = StringIO()
    export_graphviz(dtc, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())


if __name__ == '__main__':
    # trainPath = 'F:/University of Waterloo/Winter 2019/CS 680/Assignments/assign_1/wdbc-train.csv'
    # testPath = 'F:/University of Waterloo/Winter 2019/CS 680/Assignments/assign_1/wdbc-test.csv'
    trainPath = str(sys.argv[1])
    testPath = str(sys.argv[2])
    max_depth_range = 10
    criteria_dt = 'entropy'  # 'gini'
    plot_score(trainPath, testPath, max_depth_range, criteria_dt)
    # max_depth_final = 3
    # final_model(trainPath, testPath, max_depth_final)
