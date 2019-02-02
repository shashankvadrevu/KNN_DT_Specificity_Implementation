import sys
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as NNC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold as KF
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import pprint
import matplotlib.pyplot as plt


pd.set_option('display.expand_frame_repr', False)


def load_data(trainPath, testPath):

    train_df = pd.read_csv(trainPath, delimiter=",", header=None, index_col=0)
    test_df = pd.read_csv(testPath, delimiter=",", header=None, index_col=0)

    # print(len(test_df.groupby([1]).groups['M']))
    # B = 250, M = 149 --> train
    # B = 63, M = 107 --> test
    return train_df, test_df


def preProcessing(trainPath, testPath):

    train_df, test_df = load_data(trainPath, testPath)

    simpleScaler = MinMaxScaler()
    train_df[train_df.columns.drop([1])] = simpleScaler.fit_transform(
        train_df[train_df.columns.drop([1])])
    test_df[test_df.columns.drop([1])] = simpleScaler.transform(
        test_df[test_df.columns.drop([1])])

    return train_df, test_df


def knn_model(trainPath, testPath, neighbors_range):
    train_df, test_df = preProcessing(trainPath, testPath)
    knn_score_cv = {}
    for each in range(1, neighbors_range + 1):
        knn = NNC(n_neighbors=each, leaf_size=30, p=2)
        cv_kf = KF(n_splits=10, shuffle=True, random_state=123)
        split_cv = cv_kf.split(train_df)
        cross_val_scores = {}
        for i, eachSplit in enumerate(split_cv):
            train_cv = train_df.iloc[eachSplit[0]]
            test_cv = train_df.iloc[eachSplit[1]]
            knn_cv = knn.fit(train_cv[train_cv.columns.drop(
                [1])], train_cv[1])
            # y_pred = cross_val_predict(knn_cv, test_cv[test_cv.columns.drop(
            #     [1])], test_cv[1], cv=2)
            y_pred = knn_cv.predict(test_cv[test_cv.columns.drop(
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

        knn_score_cv[each] = {"recall_avg_cv": recall_avg, 'precision_avg_cv': precision_avg,
                              'sensitivity_avg_cv': sensitivity_avg, 'specificity_avg_cv': specificity_avg}

    knn_train_test_scores = {}
    for each in range(1, neighbors_range + 1):
        knn = NNC(n_neighbors=each, leaf_size=30, p=2)
        train2_df, test2_df = preProcessing(trainPath, testPath)
        knn_tt = knn.fit(train2_df[train_df.columns.drop(
            [1])], train2_df[1])
        y_pred2 = knn_tt.predict(test2_df[test2_df.columns.drop(
            [1])])
        tn2, fp2, fn2, tp2 = confusion_matrix(test2_df[1], y_pred2).ravel()
        recall_tt = round(((tp2 / (tp2 + fn2)) * 100), 2)
        sensitivity_tt = recall_tt
        precision_tt = round(((tp2 / (tp2 + fp2)) * 100), 2)
        specificity_tt = round(((tn2 / (fp2 + tn2)) * 100), 2)

        knn_train_test_scores[each] = {"recall_model": recall_tt, 'precision_model': precision_tt,
                                       'sensitivity_model': sensitivity_tt, 'specificity_model': specificity_tt}

    return knn_score_cv, knn_train_test_scores


def plot_score(trainPath, testPath, neighbors_range):
    knn_score_cv, knn_train_test_scores = knn_model(
        trainPath, testPath, neighbors_range)
    recall_cvavg_lst = []
    precision_cvavg_lst = []
    specificity_cvavg_lst = []
    for k, v in knn_score_cv.items():
        recall_cvavg_lst.append(v['recall_avg_cv'])
        precision_cvavg_lst.append(v['precision_avg_cv'])
        specificity_cvavg_lst.append(v['specificity_avg_cv'])

    cv_plt = plt.figure()
    plt.plot(knn_score_cv.keys(), recall_cvavg_lst)
    plt.plot(knn_score_cv.keys(), precision_cvavg_lst)
    plt.plot(knn_score_cv.keys(), specificity_cvavg_lst)
    # plt.plot(knn_score_cv.keys(), specificity_model_lst)
    cv_plt.suptitle(
        'KNN-50 10-Fold CrossVal Recall, Precision, Specificity', fontsize=12)
    cv_plt.legend(['Recall_CrossVal', 'Precision_CrossVal',
                   'Specificity_CrossVal'], loc='lower left')
    plt.ylim(85, 102)
    plt.ylabel('Percent (%)')
    plt.xlabel('Number of Neighbors')
    plt.show('hold')

    recall_model_lst = []
    precision_model_lst = []
    specificity_model_lst = []

    for k, v in knn_train_test_scores.items():
        recall_model_lst.append(v['recall_model'])
        precision_model_lst.append(v['precision_model'])
        specificity_model_lst.append(v['specificity_model'])

    model_plt = plt.figure()
    plt.plot(knn_score_cv.keys(), recall_model_lst)
    plt.plot(knn_score_cv.keys(), precision_model_lst)
    plt.plot(knn_score_cv.keys(), specificity_model_lst)
    # plt.plot(knn_score_cv.keys(), specificity_model_lst)
    model_plt.suptitle(
        'KNN-50 10-Fold Model Recall, Precision, Specificity', fontsize=12)
    model_plt.legend(['Recall_Model', 'Precision_Model',
                      'Specificity_Model'], loc='lower left')
    plt.ylim(85, 102)
    plt.ylabel('Percent (%)')
    plt.xlabel('Number of Neighbors')
    plt.show('hold')

    cv_model_spcf_plt = plt.figure()
    plt.plot(knn_score_cv.keys(), specificity_cvavg_lst)
    plt.plot(knn_score_cv.keys(), specificity_model_lst)
    cv_model_spcf_plt.suptitle(
        'KNN-50 10-Fold CrossVal Vs Model Specificity', fontsize=12)
    model_plt.legend(['Specificity_Model'], loc='lower left')
    plt.ylim(90, 102)
    plt.ylabel('Specificity (%)')
    plt.xlabel('Number of Neighbors')
    plt.show('hold')

    # split_cv_n = next(split_cv, None)

    # y_pred = cross_val_predict(knn_cv, train_df[train_df.columns.drop(
    #     [1])], train_df[1], cv=5)
    # conf_mat = confusion_matrix(train_df[1], y_pred)
    # print(conf_mat)
    # cv_kf = KF(n_splits=10, shuffle=True, random_state=123)
    # split_cv = cv_kf.split(train_df)
    # split_cv_n = next(split_cv, None)
    # print(train_df.iloc[split_cv_n[0]], train_df.iloc[split_cv_n[1]])
    # score_list = ['accuracy', 'precision', 'recall', ]
    # cv_scores = cross_val_score(knn_cv, train_df[train_df.columns.drop(
    #     [1])], train_df[1], cv=split_cv, scoring='confusion_matrix')
    # print(train_df.iloc[split_cv[0]], train_df.iloc[split_cv[1]]) train, test
    # NNC_model = NNC(n_neighbors=1, leaf_size=30, p=2)

    return print("Sucess: KNN Complete")


if __name__ == '__main__':
    # trainPath = sys.argv[1]
    # trainPath = 'F:/University of Waterloo/Winter 2019/CS 680/Assignments/assign_1/wdbc-train.csv'
    # testPath = 'F:/University of Waterloo/Winter 2019/CS 680/Assignments/assign_1/wdbc-test.csv'
    trainPath = str(sys.argv[1])
    testPath = str(sys.argv[2])
    neighbors_range = 50
    plot_score(trainPath, testPath, neighbors_range)
