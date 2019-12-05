library(tidyverse)
library(readr)
library(ggplot2)
library(caret)

raw_tweets <-  read_csv(unzip("data/Sentiment-Analysis-Dataset.zip"))

raw_tweets$Sentiment <- as.factor(raw_tweets$Sentiment)
raw_tweets$SentimentSource <- as.factor(raw_tweets$SentimentSource)

table(raw_tweets[,2:3])
prop.table(table(raw_tweets[, "Sentiment"]))

           