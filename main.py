# 开发时间  2023/6/7 20:36
import matplotlib
from numpy import interp
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from kmeans import *
from sklearn.model_selection import train_test_split


random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
matplotlib.use('TkAgg')

def train(X, Y):

    X, Y = shuffle(X, Y, random_state=6)
    clf = MLPClassifier((128, 64), solver='adam', alpha=1e-5, random_state=10,
                        max_iter=1000)
    A, X_test, B, Y_test = train_test_split(X, Y, test_size=0.1, random_state=65)

    kf = KFold(n_splits=5)
    print("开始训练!")
    tprs = []
    precisions = []
    AUC_list = []
    mean_fpr = np.linspace(0, 1, 100)

    acc_list = []
    p_list = []
    r_list = []
    f1_list = []
    AUPR_list = []
    i = 1
    for train_index, test_index in kf.split(A, B):

        X_train = A[train_index]
        Y_train = B[train_index]

        X_test = A[test_index]
        Y_test = B[test_index]
        clf.fit(X_train, Y_train)
        predict_value = clf.predict_proba(X_test)[:, 1]
        acc_pre = np.argmax(clf.predict_proba(X_test), axis=1)

        '''画图'''
        fpr, tpr, _ = roc_curve(Y_test, predict_value, drop_intermediate=False)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        AUC_list.append(roc_auc)

        precision, recall, _ = precision_recall_curve(Y_test, predict_value)

        precisions.append(interp(mean_fpr, recall[::-1], precision[::-1]))
        AUPR = auc(recall, precision)
        AUPR_list.append(AUPR)

        p = precision_score(Y_test, predict_value.round())
        p_list.append(p)
        r = recall_score(Y_test, predict_value.round())
        r_list.append(r)
        f1 = f1_score(Y_test, predict_value.round())
        f1_list.append(f1)
        acc = accuracy_score(Y_test, acc_pre)
        acc_list.append(acc)

        i = i + 1
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Y_prob = clf.predict_proba(X_test)[:, 1]
    # 计算 ROC-AUC
    fpr, tpr, _ = roc_curve(Y_test, Y_prob, drop_intermediate=False)
    interp(mean_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    print(f"validation:roc: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(Y_test, Y_prob)
    aupr = average_precision_score(Y_test, Y_prob)
    print(f"validation:aupr: {aupr:.4f}")
    #
    p = precision_score(Y_test, Y_prob.round())
    print(f"validation:Precision: {p:.4f}")

    r = recall_score(Y_test, Y_prob.round())
    print(f"validation:Recall: {r:.4f}")

    f1 = f1_score(Y_test, Y_prob.round())
    print(f"validation:F1 Score: {f1:.4f}")

    acc_p = np.argmax(clf.predict_proba(X_test), axis=1)
    acc = accuracy_score(Y_test, acc_p)
    print(f"validation:Accuracy: {acc:.4f}")

    auc_l = np.array(AUC_list)
    aupr_l = np.array(AUPR_list)

    p_l = np.array(p_list)
    r_l = np.array(r_list)
    f1_l = np.array(f1_list)
    acc_l = np.array(acc_list)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_precision = np.mean(precisions, axis=0)
    mean_precision[-1] = 0

    # 打印对应的均值和标准差
    print('AUROC = %.4f +- %.4f | AUPR = %.4f +- %.4f' % (auc_l.mean(), auc_l.std(), aupr_l.mean(),
                                                          aupr_l.std()))
    print('Precision = %.4f +- %.4f' % (p_l.mean(), p_l.std()))
    print('Recall = %.4f +- %.4f' % (r_l.mean(), r_l.std()))
    print('F1_score = %.4f +- %.4f' % (f1_l.mean(), f1_l.std()))
    print('Accuracy = %.4f +- %.4f' % (acc_l.mean(), acc_l.std()))


if __name__ == '__main__':

    ConnectData = np.loadtxt('data/known_DM_association.txt', dtype=int) - 1
    UnknownData  = np.loadtxt('data/unknown_DM_association.txt', dtype=int) - 1

    A = np.zeros((627, 142), dtype=float)
    for i in range(1152):
        A[ConnectData[i, 0], ConnectData[i, 1]] = 1

    drug_svd = np.genfromtxt('data/drug_svd_feature.csv', delimiter=',')
    drug_nmf = np.genfromtxt('data/drug_nmf_feature.csv', delimiter=',')
    drug_gan_gcn = np.genfromtxt('data/drug_gan_feature_gcn.csv', delimiter=',')

    mic_svd = np.genfromtxt('data/mic_svd_feature.csv', delimiter=',')
    mic_nmf = np.genfromtxt('data/mic_nmf_feature.csv', delimiter=',')  # 64
    mic_gan_gcn = np.genfromtxt('data/mic_gan_feature_gcn.csv', delimiter=',')

    drug_fea = np.hstack((drug_gan_gcn, drug_svd, drug_nmf))
    mic_fea = np.hstack((mic_gan_gcn, mic_svd, mic_nmf))

    Zheng = get_Zheng(ConnectData, drug_fea, mic_fea)
    Fu = get_kmeans_fu(UnknownData, drug_fea, mic_fea)
    feature, labels = get_feature(Zheng, Fu)
    train(feature, labels)