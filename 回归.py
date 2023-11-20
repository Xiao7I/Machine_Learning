import numpy as np
import matplotlib.pyplot as plt
from numpy import mat
from sklearn import datasets
from numpy import *


# 线性回归首先是对数据进行处理，将数据分割成特征数据和目标数据，并将这两种数据提取出特征矩阵和目标矩阵
# 对特征矩阵和目标矩阵进行计算，算出使代价函数最小的回归系数，具体数学公式自行查阅
# 最后进行绘图


# 加载处理数据
def loadDates(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    datamat = []
    labelmat = []
    f = open(filename)
    for line in f.readlines():
        temp = []
        curline = line.strip().split('\t')
        # 将该行以\t为标识分割，并且除去分割后每一部分的前后空格 strip()方法用于移除字符串首尾指定的字符串，没有参数则默认空格
        for i in range(numFeat):
            temp.append(float(curline[i]))
        datamat.append(temp)
        labelmat.append(float(curline[-1]))
    # print(type(datamat), type(labelmat))
    return datamat, labelmat


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # print(type(xMat), type(yMat))
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# xArr, yArr = loadDates("C:\\Users\\ASUS\Desktop\\MachineLearning\\data.txt")
# ws = standRegres(xArr, yArr)
# print(ws)

def regression1():
    # xArr, yArr = loadDates("C:\\Users\\ASUS\Desktop\\MachineLearning\\data.txt")
    xArr, yArr = loadDates("C:\\Users\\Admin\Desktop\\MachineLearning\\data.txt")
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    print(xMat[:, 1].flatten())
    print(yMat.T[:, 0].flatten())
    x = xMat[:, 1].flatten().A[0]
    y = yMat.T[:, 0].flatten().A[0]
    # flatten()方法是用来降维的，默认按行展开，例如n行1列矩阵转成1行n矩阵   .A[0]是将矩阵变换为数组
    ax.scatter(x, y, c='r', marker='o')
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


regression1()
