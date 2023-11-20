import numpy as np
from numpy import *


# A = np.matrix('1 2; 3 4')
# A_1 = array([[1, 2], [3, 4]])

A = array([[1, 0 + 1j, 1], [0 - 1j, 0, 0 - 2j], [1, 0 + 2j, 0]])
eigenvalue, featurevector = np.linalg.eig(A)
print(eigenvalue)
print(featurevector)
# B = np.linalg.inv(A)
# print(B)
# C = array([[1, 2, 3], [-1, 0, 3], [2, 1, 5]])
# print(B * C)
# D = array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
# print(B * C * D)
# print(A)
# # 矩阵的转置
# B = A.T
# B_1 = np.transpose(A_1)
# print(B_1)
# print(B)
# # 矩阵的逆
# C = A.I
# C_1 = np.linalg.inv(A)
# print(C_1)
# print(C)
# # 矩阵的秩和维数
# print(A.ndim)
# print(np.linalg.matrix_rank(A))
# # 零矩阵
# D = zeros((3, 3))
# print(D)
# # 每个元素为1的矩阵
# E = ones((3, 3))
# print(E)
# # 单位阵
# F = eye(3)
# # 矩阵加法
# print(A + B)
# # 矩阵乘法
# print(dot(A, B))
# print(A * B)
# # 矩阵的秩
# print(np.linalg.det(F))
# # 矩阵的伪逆
# G = np.matrix("1 1; 0 0")
# G_1 = np.linalg.pinv(G)
# print(G_1)
# print(G[1])
# A_inv = np.linalg.inv(A_1)
# print(A_inv)
# print(A_1[0][0])
# T = array([[1], [2], [3]])
# print(T)
# print(1/T)
# print(log(T))
# print(log(T)/log(2))
# T_1 = array([[-1], [-2], [-3]])
# print(abs(T_1))
# C_1 = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# F_1 = C_1 + F
# print(F_1)
# print(F_1[0][0])
# print(F_1 + 1)
# print(F_1 * 2)





