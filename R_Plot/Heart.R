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
require("ggplot2")
# 用ggplot2画直方图
if (FALSE) {
  g <- ggplot(population, aes(x=population2020)) +
    geom_histogram(binwidth = 5000000, color='black', fill='white') +
    ggtitle("China Population Histogram")
  g
}
# 为数据增加新的列，来表示人口从2010到2020是否增长了
population$popChange <- population$population2020 - population$population2010
population$increasePop <- ifelse(population$popChange > 0, "positive", "negative")
# 用ggplot2画箱线图
if (FALSE) {
  ggplot(population, aes(x=factor(increasePop), population2010)) +
    geom_boxplot() +
    coord_flip() +
    ggtitle('Population grouped by positive or negative change')
}
# 用ggplot2画线图
if (FALSE) {
  g <- ggplot(population, aes(x=reorder(ChineseName, population2020), y=population2020, group=1)) +
    geom_line() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  g
}
# 用ggplot2画条形图
if (FALSE) {
  g <- ggplot(population, aes(x=reorder(ChineseName, population2020), y=population2020, group=1)) +
    geom_col() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  g
}
# 用ggplot2画一个条形图，其中用不同颜色表示另一个属性（人口增长率）
population$percentChange <- population$popChange/population$population2020 * 100
if (FALSE) {
  g <- ggplot(population, aes(x=reorder(ChineseName, population2020), y=population2020, fill=percentChange)) +
    geom_col() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  g
}
# 选两列画散点图，并加标签
if (TRUE) {
  ggplot(population, aes(x=popChange, y=percentChange)) +
    geom_point(aes(size=population2020, color=population2020)) +
    geom_text(aes(label=ChineseName), size=4, hjust=1, vjust=-1)  # 给散点加label
  # hjust, vjust: adjusting the horizontal and vertical position of the text
}
if (TRUE) {
  minPerChange <- 1
  minPopChange <- 100000
  population$keyProvince <- population$popChange>minPopChange & population$percentChange > minPerChange
  minLabel <- format(min(population$population2020), big.mark = ",", trim = TRUE)
  maxLabel <- format(max(population$population2020), big.mark = ",", trim = TRUE)
  medianLabel <- format(median(population$population2020), big.mark = ",", trim = TRUE)
  g <- ggplot(population, aes(x=popChange, y=percentChange)) +
    geom_point(aes(size=population2020, color=population2020, shape=keyProvince)) +
    geom_text(data = poulation[population$popChange > minPopChange & population$percentChange > minPerChange,],
              aes(label=ChineseName, hjust=1, vjust=-1)) +
    scale_color_continuous(name="Pop", breaks = with(population, c(
      min(population2020), median(population2020), max(population2020))),
    labels = c(minLabel, medianLabel, maxLabel), low = "white", high = "black")
}