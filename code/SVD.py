import numpy as np
import pandas as pd

DFdata = pd.read_csv('association_matrix.csv', header=None)  # 0,1矩阵
ArrDate = np.array(DFdata)

U,s,V = np.linalg.svd(ArrDate)


# 选取前 173 维作为特征向量
SVD_1 = np.zeros((627, 128))
for i in range(627):
    for j in range(64):
        SVD_1[i,j] = U[i, j]

SVD_2 = np.zeros((142,128))
for i in range(142):
    for j in range(64):
        SVD_2[i,j] = V[j,i]

print(SVD_1.shape, SVD_2.shape)
np.savetxt('drug_svd_feature.csv', SVD_1, delimiter=',')
np.savetxt('mic_scd_feature.csv', SVD_2, delimiter=',')

