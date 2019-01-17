# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:48:50 2019
@author: wmin_
"""
import operator
import numpy as np
from os import listdir


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group, labels = createDataSet()

print('group:', group)
print('labels:', labels)  # 输出数值

def img2vector(filename):
    # 创建向量
    returnVect = np.zeros((1, 1024))
    # 打开数据文件，读取每行内容
    fr = open(filename)
    for i in range(32):
        # 读取每一行
        lineStr = fr.readline()
        # 将每行前 32 字符转成 int 存入向量
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
            
    return returnVect

tmp = img2vector('./digits/testDigits/0_1.txt')

# =============================================================================
# 计算已知类别数据集中的点与当前点之间的距离；
# 按照距离递增次序排序；
# 选取与当前点距离最小的 k 个点；
# 确定前 k 个点所在类别的出现频率；
# 返回前 k 个点出现频率最高的类别作为当前点的预测分类。

# trainingDigits：训练数据，1934 个文件，每个数字大约 200 个文件。
#testDigits：测试数据，946 个文件，每个数字大约 100 个文件。
# =============================================================================
def classify(inX, dataSet, labels, k):   
    """
    参数: 
    - inX: 用于分类的输入向量
    - dataSet: 输入的训练样本集
    - labels: 样本数据的类标签向量
    - k: 用于选择最近邻居的数目
    """    
    # 获取样本数据数量
    dataSetSize = dataSet.shape[0]  #1934

    # 矩阵运算，计算测试数据与每个样本数据对应数据项的差值
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # tile()矩阵复制，同型dataset

    # sqDistances 上一步骤结果平方和
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)

    # 取平方根，得到距离向量
    distances = sqDistances**0.5

    # 按照距离从低到高排序
    # argsort()将x中的元素从小到大排列，提取其对应的index(索引)，
    # 然后输出将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    sortedDistIndicies = distances.argsort() 
    classCount = {}

    # 依次取出最近的样本数据
    for i in range(k):
        # 记录该样本数据所属的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 对类别出现的频次进行排序，从高到低
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)    
    #operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号
    
    # 返回出现频次最高的类别
    return sortedClassCount[0][0]


group, labels = createDataSet()
tmp = classify([0, 0], group, labels, 3)



def handwritingClassTest():
    # 样本数据的类标签列表
    hwLabels = []

    # 样本数据文件列表
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)

    # 初始化样本数据矩阵（M*1024）
    trainingMat = np.zeros((m, 1024))

    # 依次读取所有样本数据到数据矩阵
    for i in range(m):
        # 提取文件名中的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        # 将样本数据存入矩阵
        trainingMat[i, :] = img2vector(
            'digits/trainingDigits/%s' % fileNameStr)

    # 循环读取测试数据
    testFileList = listdir('digits/testDigits')

    # 初始化错误率
    errorCount = 0.0
    mTest = len(testFileList)

    # 循环测试每个测试数据文件
    for i in range(mTest):
        # 提取文件名中的数字
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        # 提取数据向量
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)

        # 对数据文件进行分类
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)

        # 打印 K 近邻算法分类结果和真实的分类
        print("测试样本 %d, 分类器预测: %d, 真实类别: %d" %
              (i+1, classifierResult, classNumStr))

        # 判断K 近邻算法结果是否准确
        if (classifierResult != classNumStr):
            errorCount += 1.0

    # 打印错误率
    print("\n错误分类计数: %d" % errorCount)
    print("\n错误分类比例: %f" % (errorCount/float(mTest)))
    
def main():
	handwritingClassTest()    
   
if __name__ == "__main__":
    main()