from numpy import *
import numpy as np
import operator
from os import listdir
import time

#分类器,k-近邻算法
def classify0(inX, dataSet, labels, k):
    #求出样本集的行数，也就是labels标签的数目
    dataSetSize = dataSet.shape[0]
    #构造输入值和样本集的差值矩阵
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    #计算欧式距离
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #求距离从小到大排序的序号
    sortedDistIndicies = distances.argsort()
    #对距离最小的k个点统计对应的样本标签
    classCount={}
    for i in range(k): #选择距离最小的k个点
        #取第i+1邻近的样本对应的类别标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #以标签为key，标签出现的次数为value将统计到的标签及出现次数写进字典
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #对字典按value从大到小排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True) #排序
    #返回排序后字典中最大value对应的key
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) #创建数据集
    labels = ['A','A','B','B'] #创建标签
    return group,labels

#解析文本记录
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #得到文件的行数
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #创建以零填充的矩阵，为了简化处理，将该矩阵的另一维度设置为固定值3，可按照自己的实际需求增加相应的代码以适应变化的输入值
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()#截取掉所有的回车字符
        listFromLine = line.split('\t') #然后使用tab字符\t将上一步得到的整行数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3] #选取前3个元素，将它们存储到特征矩阵中
        if(listFromLine[-1] == 'largeDoses'): #Python语言可使用索引值-1表示列表中的最后一列元素，利用该负索引，可方便地将列表的最后一列存储到向量classLabelVector中
            classLabelVector.append(3)# listFromLine[-1] = '3'
        elif (listFromLine[-1] == 'smallDoses'):
            classLabelVector.append(2)#listFromLine[-1] = '2'
        else:
            classLabelVector.append(1)#listFromLine[-1] = '1'
      #  classLabelVector.append(float(listFromLine[-1]))#必须明确地通知解释器，告诉它列表中存储的元素值为整型，否则Python语言会将这些元素当做字符串处理
        index += 1

    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) #每列最小值
    maxVals = dataSet.max(0) #每列最大值
    ranges = maxVals - minVals #函数计算可能的取值范围
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1)) #tile将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = normDataSet/np.tile(ranges, (m,1)) #特征值相除，为了归一化特征值，必须使用当前值减去最小值，然后除以取值范围；在某些数值处理软件包，/可能意味着矩阵除法，NumPy库中，
    return normDataSet,ranges,minVals            #矩阵除法需要使用函数linalg.solve(matA,matB)

#测试算法：作为完整程序验证分类器
def datingClassTest():
    hoRatio = 0.1
    #从文件中读取数据并将其转换为归一化特征值
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    #计算测试向量的数量，决定了normMat向量中哪些数据用于测试，哪些数据用于分类器的训练样本
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #将测试数据和训练数据输入到分类器中
        #normMat[i,:] 取出第i行的所有数据
        #normMat[numTestVecs:m,:]取出numTestVecs之后到m的每行数据
        #datingLabels[numTestVecs:m]取出numTestVecs之后到m的每行的标签
        #k值为3
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        #如果错误不一致，则错误数加1
        if(classifierResult != datingLabels[i]):errorCount += 1.0
    #计算出错误率并输出结果
    print("the total error rate is:%f"%(errorCount/float(numTestVecs)))

#预测函数：
"""
def classifyPerson():
    resultList = ['不喜欢', '有点喜欢', '非常喜欢']
    percentTats = float(input("玩视频游戏所耗时间百分比： "))
    ffMiles = float(input("每年获得飞行常客里程数： "))
    iceCream = float(input("每周消费冰激凌公升数： "))
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([percentTats, ffMiles, iceCream])
    norminArr = (inArr - minVals) / ranges
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    print("你可能对这个人：", resultList[classifierResult-1])
"""
def classifyPerson():
	#输出结果
	resultList = ['不喜欢','有些喜欢','非常喜欢']
	#三维特征用户输入
	precentTats = float(input("玩视频游戏所耗时间百分比:"))
	ffMiles = float(input("每年获得的飞行常客里程数:"))
	iceCream = float(input("每周消费的冰激淋公升数:"))
	#打开的文件名
	filename = "datingTestSet.txt"
	#打开并处理数据
	datingDataMat, datingLabels = file2matrix(filename)
	#训练集归一化
	normMat, ranges, minVals = autoNorm(datingDataMat)
	#生成NumPy数组,测试集
	inArr = np.array([ffMiles, precentTats, iceCream])
	#测试集归一化
	norminArr = (inArr - minVals) / ranges
	#返回分类结果
	classifierResult = classify0(norminArr, normMat, datingLabels, 3)
	#打印结果
	print("你可能%s这个人" % (resultList[classifierResult-1]))

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    cpu_start = time.time()
    print('start:%f' % cpu_start)
    hwLabels = []
    #获取目录内容
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros([m,1024])
    for i in range(m):
        #从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类器返回值为:%d, 实际值为:%d" %(classifierResult, classNumStr))
        if(classifierResult != classNumStr): errorCount += 1.0
    print("\n错误总数为： %d" % errorCount)
    print("\n总错误率为： %f" %(errorCount/float(mTest)))
    cpu_end = time.time()
    print('end:%f' % cpu_end)
    print("total time: %f S" % (cpu_end - cpu_start))
if __name__ == '__main__':
    # datingClassTest()
    #classifyPerson()
   # testVector=img2vector('testDigits/0_13.txt')
    #print(testVector[0,0:31])
   # print(testVector[0,32:63])
    handwritingClassTest()