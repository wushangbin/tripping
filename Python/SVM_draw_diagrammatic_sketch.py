# Title     : SVM - draw the diagrammatic sketch
# Objective : draw the diagrammatic sketch for SVM. Note that it is not the code for SVM
# Created by: Wu Shangbin
# Created on: 2021/12/8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
 
sampleNo = 20
 
# 二维正态分布
gaussian1 = np.random.normal(loc=[1,1], scale=0.4, size=(sampleNo,2))
gaussian2 = np.random.normal(loc=[-1,-1], scale=0.4, size=(sampleNo,2))
data_x = np.concatenate((gaussian1, gaussian2))
data_y = [0 for i in range(sampleNo)] + [1 for i in range(sampleNo)] 
data_y = np.array(data_y)
scaler = StandardScaler()
data_x = scaler.fit_transform(data_x)
# -- 
svc = LinearSVC(C=1.0)
model = svc.fit(data_x, data_y)
# --
w = svc.coef_[0]
k = - w[0]/w[1]
xx = np.linspace(-2, 2)
yy = k * xx - (svc.intercept_[0])/w[1]
# 绘制散点图
plt.plot(xx, yy)
plt.plot(gaussian1[:,0],gaussian1[:,1],'+', color='r')
plt.plot(gaussian2[:,0],gaussian2[:,1],'o', color='g')
plt.show()


