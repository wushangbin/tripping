# Title     : Word Perfect
# Objective : 进行文本分析
# Created by: Wu Shangbin
# Created on: 2021/6/3
# 参考：Jeffrey S. SALTZ, JEFFREY M. STANTON "An Introduction to Data Science"
# --- 从本地文件读取文本：
if (TRUE) {
  setwd("E:/Project/tripping/R_Plot/Data")
  textfile <- "GB.txt"
  # 一行一行地读取文本数据
  # 方法1：
  text <- scan(textfile, character(0), sep='\n')
  #方法2： text <- readLines(textfile)
}
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
  web <- getURL(url)
  doc<-htmlTreeParse(web,encoding="UTF-8", error=function(...){}, useInternalNodes = TRUE,trim=TRUE)
  text <- unlist(xpathApply(doc, '//p', xmlValue))
  head(text, 3)  # 在text中，每个元素是一行字符串
}
library("tm")
words.vec <- VectorSource(text)
words.corpus <- Corpus(words.vec)
# 第二个参数里的content_transformer貌似加不加都一样的,这里对数据进行预处理
words.corpus <- tm_map(words.corpus, content_transformer(tolower))
words.corpus <- tm_map(words.corpus, content_transformer(removePunctuation))
words.corpus <- tm_map(words.corpus, removeNumbers)
words.corpus <- tm_map(words.corpus, removeWords, stopwords("english"))  # 删除stop words
tdm <- TermDocumentMatrix(words.corpus)
tdm
# 输出解读：
#<<TermDocumentMatrix (terms: 578, documents: 37)>> 一共有578个term，37个documents（37行）
#Non-/sparse entries: 934/20452  在term-documents构成的578*37矩阵中，非空的只有934个元素
#Sparsity           : 96%  有96%的元素是空值
#Maximal term length: 16  一个term并非是一个单词，可能是一个词组，这里是最大term长度（16个字母）
#Weighting          : term frequency (tf)
#
m <- as.matrix(tdm)
wordCounts <- rowSums(m)
wordCounts <- sort(wordCounts, decreasing = TRUE)
head(wordCounts)
cloudFrame <- data.frame(word=names(wordCounts), freq=wordCounts)
# -- 绘制wordcloud
library("wordcloud")
wordcloud(cloudFrame$word, cloudFrame$freq)

# -- Sentiment Analysis
setwd("../Data/opinion-lexicon-English")
pos <- "positive-words.txt"
neg <- "negative-words.txt"
p <- scan(pos, character(0), sep="\n")
n <- scan(neg, character(0), sep="\n")
p <- p[-1:-29]  # 文件的开头有些别的东西，要去掉
n <- n[-1:-30]
head(p, 10)
head(n, 10)
# --
totalWords <- sum(wordCounts)  # 总的单词数目
words <- names(wordCounts)  # 都有哪些单词
matched <- match(words, p, nomatch=0)  # 返回一个矩阵，表示words中的每一个单词是p中的第几个单词
mCounts <- wordCounts[which(matched != 0)]
length(mCounts)  # 这篇文章中出现的positive的单词的种类个数（there were 40 unique positive words）
nPos <- sum(mCounts)
nPos  # 共有58个positive words
# -- 下面对negative words做同样的操作，统计unique negative words的个数和negative words的个数
matched <- match(words, n, nomatch = 0)
nCounts <- wordCounts[which(matched != 0)]
nNeg <- sum(nCounts)
length(nCounts)
nNeg
# --
totalWords <- length(words)
ratioPos <- nPos / totalWords
ratioNeg <- nNeg / totalWords
ratioPos
ratioNeg