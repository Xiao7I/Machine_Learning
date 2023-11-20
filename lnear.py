from numpy import mat
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# def loadDateSet():
x, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=4)  # 产生100个样本，样本的特征属性和标签都是一，噪声为1.


# numFeat = 100
# dateMat = []
# labelMat = []
# print(x)
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws
# theta = 10
# alpha = 0.1
# def gradient_descent(xArr, yArr, theta, alpha):
#     xMat = mat(xArr)
#     yMat = mat(yArr)
#     theta = theta - alpha * (yMat - theta * xMat).T * xMat
#     return theta
#
# xMat = mat(x)
# yMat = mat(y)
# print(xMat[:,].flatten().A[0],yMat.T[:, 0].flatten().A[0])
def regression1():
    xMat = mat(x)
    yMat = mat(y)
    ws = standRegres(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,0].flatten().A[0], yMat.T[:, 0].flatten().A[0])  # scatter 的x是xMat中的第二列，y是yMat的第一列
    xCopy = xMat.copy()
    xCopy.sort(0)
    print(xCopy)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 0], yHat)
    plt.show()


regression1()
# plt.scatter(x, y)#散点的形式输出在图上
# plt.show()
