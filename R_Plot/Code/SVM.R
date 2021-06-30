# Title     : Support Vector Machine with R
# Objective : Using R to realize support vector machine
# Created by: Wu Shangbin
# Created on: 2021/6/24
# Reference : Jeffrey S. SALTZ, JEFFREY M. STANTON "An Introduction to Data Science"
# Data used : ewlett-Packard公司的雇员收到的邮件:：https://archive.ics.uci.edu/ml/datasets/Spambase
# Pkgs used : "kernlab",
# install.packages("kernlab")
library(kernlab)
data(spam) # 使用Hewlett-Packard公司的雇员收到的邮件作为数据集，任务是：区分普通邮件和垃圾邮件
# 关于数据，可参考此链接：https://archive.ics.uci.edu/ml/datasets/Spambase
# 简单地查看一下数据
str(spam)
dim(spam) # 4601*58
table(spam$type)
#---划分训练集与测试集
randIndex <- sample(1:dim(spam)[1])
# 创造了一个从1到4601的随机序列
summary(randIndex)
length(randIndex)
head(randIndex)
cutPoint2_3 <- floor(2*dim(spam)[1] / 3)
cutPoint2_3

trainData <- spam[randIndex[1:cutPoint2_3], ]
testData <- spam[randIndex[(cutPoint2_3+1):dim(spam)[1]], ]
print(dim(spam))
print(dim(trainData))
print(dim(testData))
svmOutput <- ksvm(type ~., data=trainData, kernel="rbfdot", kpar="automatic", C=5, cross=3, prob.model=TRUE)
# parameters:
# type ~. : 以trainData中其他的列作为特征，type作为label
# kernel="rbfdot" : 这个核函数的功能是，新加一个特征：点*到原点距离
# C=5 : C越小会允许更多的分类错误，C越大，越要求超平面严格分割
# cross=3 : cross-validation
print(svmOutput)
hist(alpha(svmOutput)[[1]])
# 这个直方图中，横轴是support vector。其最大值是我们刚刚设置的C=5，越大的值表示越难分类的样本
# 如果我们将C的值取得更大：
svmOutput <- ksvm(type ~., data=trainData, kernel="rbfdot", kpar="automatic", C=50, cross=3, prob.model=TRUE)
svmOutput
# 会发现training error更小了，但是cross validation error更大了，也就是说，有过拟合
hist(alpha(svmOutput)[[1]], main="Support Vector Histogram with C=50", xlab="Support Vector Values")
# 看看哪些样本是易分类的:(支持向量的值比较小)
alphaindex(svmOutput)[[1]][alpha(svmOutput)[[1]] < 0.05]
trainData[54, ]
# 看看难分类的垃圾邮件：
alphaindex(svmOutput)[[1]][alpha(svmOutput)[[1]] == 50]
trainData[15, ]
svmPred <- predict(svmOutput, testData, type = "votes")
print(dim(testData))
print(dim(svmPred))
compTable <- data.frame(testData[,58], svmPred[1,])
# confusion matrix
table(compTable)
