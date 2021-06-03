# Title     : Word Perfect
# Objective : 进行文本分析
# Created by: Wu Shangbin
# Created on: 2021/6/3
setwd("E:/Project/tripping/R_Plot/Data")
textfile <- "GB.txt"
# 一行一行地读取文本数据
# 方法1：
text <- scan(textfile, character(0), sep='\n')
#方法2： text <- readLines(textfile)
# --- 从网页读取文本
library(XML)  # 应注意XML不支持HTTPS协议，HTTPS应先用RCurl中的getURL载入数据，或直接把HTTPS改为HTTP
# 处理HTTP
if (FALSE) {
  textLocation <- URLencode("http://www.historyplace.com/speeches/bush-war.htm")
  doc.html <- htmlTreeParse(textLocation, useInternal=TRUE)
  text <- unlist(xpathApply(doc.html, '//p', xmlValue))
  head(text, 3)
}
# 如果处理的是HTTPS，可如下处理：
if (TRUE) {
  library(RCurl)
  url <- "https://www.jianshu.com/p/48d758ce62b4"
  web <- getURL(textLocation)
  doc<-htmlTreeParse(web,encoding="UTF-8", error=function(...){}, useInternalNodes = TRUE,trim=TRUE)
  text <- unlist(xpathApply(doc, '//p', xmlValue))
  head(text, 3)
}