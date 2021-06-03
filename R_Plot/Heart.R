# Title     : Draw your Heart
# Objective : The mission is over, and I'm exploring further
# Created by: Wu Shangbin
# Created on: 2021/5/28
# 这是我在简书上总结画图时使用的代码

# 检查一个包是否安装，如果没安装，那就安装并引入。
EnsurePackage <- function(x) {
  x <- as.character(x)
  if (!require(x, character.only=TRUE)) {
    install.packages(pkgs = x, repos="http://cran.r-project.org")
    require(x, character.only=TRUE)
  }
}
data  = read.csv("Data/heart.csv")
#hist(data$fbs)
print(names(data))
# 相关性计算
if (FALSE) {
  library(caret)
  model <- train(target~age+sex+cp+chol+trestbps, data=data, method='glm', family='binomial')
  print(model)
}
# 画联合概率密度分布图
if (FALSE) {
  library(ggpubr)
  data$sex <- as.factor(data$sex) # 先把sex转化成factor，不然R会处理为整型的0，1
  ggscatterhist(
    data,  x ='chol', y = 'trestbps',
    shape=21,color ="black",fill= "sex", size =3, alpha = 0.8,
    palette = c("#00AFBB", "#E7B800", "#FC4E07"),
    margin.plot =  "density",
    margin.params = list(fill = "sex", color = "black", size = 0.2),
    legend = c(0.9,0.15),
    ggtheme = theme_minimal()) +
    theme(axis.title.x =element_text(size=20), axis.title.y=element_text(size=20))
}
setwd("E:/Project/tripping/R_Plot")
population = read.csv("Data/China_Population.csv")
# 画直方图和条形图
if (FALSE) {
  hist(population$population2020)
  barplot(population$population2020, names.arg = population$ChineseName, las=2)
}
# 用ggplot2画直方图
if (FALSE) {
  require("ggplot2")
  g <- ggplot(population, aes(x=population2020)) +
    geom_histogram(binwidth = 5000000, color='black', fill='white') +
    ggtitle("China Population Histogram")
  g
}