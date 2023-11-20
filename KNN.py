import matplotlib.pyplot as plt
from numpy import *
from os import listdir


def loadData(filename):
    f = open(filename)
    number_lines = len(f.readlines())
    returnMat = zeros((number_lines, 3))
    labelMat = []
    f.close()
    f = open(filename)
    index = 0
    # print(type(returnMat))
    # print(type(labelMat))
    for line in f.readlines():
        line_information = line.strip().split('\t')
        returnMat[index, :] = line_information[0:3]
        labelMat.append(int(line_information[-1]))
        index += 1
    return returnMat, labelMat


# datingDataMat, datingLabels = loadData("C:/Users/ASUS\Desktop/MachineLearning/datingTestSet2.txt")


def paintPicture():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    datingDataMat, datingLabels = loadData("C:/Users/Admin/Desktop/MachineLearning/datingTestSet2.txt")
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()


def autoNorm(dataSet):
    minvals = dataSet.min(0)
    maxvals = dataSet.max(0)
    # min(0)返回该矩阵中每一列的最小值
    # min(1)返回该矩阵中每一行的最小值
    # max(0)返回该矩阵中每一列的最大值
    # max(1)返回该矩阵中每一行的最大值
    # print(minvals)
    # print(maxvals)
    ranges = maxvals - minvals
    # print(ranges)
    normdataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # shape[0]返回行数， shape[1]返回列数
    normdataSet = dataSet - tile(minvals, (m, 1))
    normdataSet = normdataSet / tile(ranges, (m, 1))
    return normdataSet, ranges, minvals


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqdiffMat = diffMat ** 2
    sqDistance = sqdiffMat.sum(axis=1)
    # 当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
    # 当axis为1时,是压缩列,即将每一行的元素相加,将矩阵压缩为一列
    distance = sqDistance ** 0.5
    sortedDistance = distance.argsort()
    #     将距离从小到大排序， 返回一个列表，里面是排序后元素在未排序前所对应的索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistance[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #   排序并返回出现最多的那个类型
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # sorted(iterable, key=None, reverse=False) 其中参数说明：iterable：可迭代对象
    # key：通过这个参数可以自定义排序逻辑
    # reverse：指定排序规则，True为降序，False为升序（默认）。
    # 字典的items方法作用：是可以将字典中的所有项，以列表方式返回。字典不是可迭代对象
    # 例如: dict1 = {'name': 'Tom', 'age': 20, 'gender': '男'}
    # print(dict1.items())  # dict_items([('name', 'Tom'), ('age', 20), ('gender', '男')])
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    # 例如: a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    # b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
    #     return sortedClassCount[0][0]
    #   利用max函数直接返回字典中value最大的key
    maxClassCount = max(classCount, key=classCount.get)
    # 按照value值来查找最大值并返回key
    return maxClassCount


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = loadData("C:/Users/Admin\Desktop/MachineLearning/datingTestSet2.txt")
    normMat, ranges, minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(hoRatio * m)
    print("numTestVecs = ", numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs: m, :], datingLabels[numTestVecs: m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


# datingClassTest()


# 约会网站预测函数
def classifyPerson():
    resultlist = ["not at all", "in small does", "in large dose"]
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = loadData("C:/Users/Admin\Desktop/MachineLearning/datingTestSet2.txt")
    normMat, ranges, minvals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifyResult = classify((inArr - minvals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this people: ", resultlist[classifyResult - 1])


# classifyPerson()

# KNN手写数字识别
# 将图像文本数据转化成向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("C:/Users/Admin\Desktop/MachineLearning/trainingDigits")
    # print(trainingFileList)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("C:/Users/Admin\Desktop/MachineLearning/trainingDigits/%s" % fileNameStr)

    testFileList = listdir("C:/Users/Admin\Desktop\MachineLearning/testDigits")
    # print(testFileList)
    error_count = 0.0
    mTest = len(testFileList)
    for j in range(mTest):
        fileNameStr = testFileList[j]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("C:/Users/Admin\Desktop/MachineLearning/testDigits/%s" % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            error_count += 1.0
    print("the total number of errors is : %d" % error_count)
    print("the total error rate is : %f" % (error_count / float(mTest)))


handwritingClassTest()
