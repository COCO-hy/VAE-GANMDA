import random
import numpy as np
from sklearn.cluster import KMeans
import torch

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

def get_kmeans_fu(UnknownData, drug_fea, mic_fea, k=7): # 7
    chu = [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 128, 144, 192, 288, 384, 576]
    shang = [576, 384, 288, 192, 144, 128, 96, 72, 64, 48, 36, 32, 24, 18, 16, 12, 9, 8, 6, 4, 3, 2]
    Fu = np.zeros((87882, 448), dtype=float)
    for i in range(87882):
        fu = []
        fu.extend(drug_fea[UnknownData[i][0]])
        fu.extend(mic_fea[UnknownData[i][1]])
        Fu[i] = fu

    kmeans = KMeans(n_clusters=chu[k], n_init=10, init='k-means++', random_state=10).fit(Fu)
    labels = kmeans.labels_

    cluster_indices = [[] for _ in range(chu[k])]
    for i, label in enumerate(labels):
        cluster_indices[label].append(i)

    num_neg_samples_per_cluster = shang[k]
    k_mean_fu_index = []

    for index in cluster_indices:
        if len(index) > 0:
            selected_index = np.random.choice(index, size=num_neg_samples_per_cluster, replace=True)
            k_mean_fu_index.extend(selected_index)
    k_means_fu = Fu[k_mean_fu_index]
    k_means_fu = np.array(k_means_fu, dtype=float)
    return k_means_fu

def get_random_fu(UnknownData, drug_fea, mic_fea):
    suiji = random.sample(list(UnknownData), 1152)
    Fu = np.zeros((1152, 448), dtype=float)
    for i in range(1152):
        fu = []
        fu.extend(drug_fea[suiji[i][0]])
        fu.extend(mic_fea[suiji[i][1]])
        Fu[i] = fu
    return Fu

def get_labels():
    labels = []
    for i in range(1152):
        labels.append([1])
    for i in range(1152):
        labels.append([0])
    labels = np.array(labels, dtype=int)
    labels = labels.flatten()
    return labels

def get_Zheng(ConnectData, drug_fea, mic_fea):
    Zheng = np.zeros((1152, 448), dtype=float)
    for i in range(1152):
        zheng = []
        zheng.extend(drug_fea[ConnectData[i, 0]])
        zheng.extend(mic_fea[ConnectData[i, 1]])
        Zheng[i] = zheng
    return Zheng

def get_feature(positives, negatives):
    feature = np.vstack((positives, negatives))
    feature = np.array(feature, dtype=float)
    labels = get_labels()
    return feature, labels





