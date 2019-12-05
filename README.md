Sentiment Analysis of Tweets
================
Kevin Bonds
12/4/2019

``` r
library(tidyverse)
library(readr)
library(ggplot2)
library(caret)
library(knitr)
```

R Markdown
----------

``` r
raw_tweets <-  read_csv(unzip("data/Sentiment-Analysis-Dataset.zip"))
```

The Data
--------

|  ItemID|  Sentiment| SentimentSource | SentimentText                                                                                                             |
|-------:|----------:|:----------------|:--------------------------------------------------------------------------------------------------------------------------|
|       1|          0| Sentiment140    | is so sad for my APL friend.............                                                                                  |
|       2|          0| Sentiment140    | I missed the New Moon trailer...                                                                                          |
|       3|          1| Sentiment140    | omg its already 7:30 :O                                                                                                   |
|       4|          0| Sentiment140    | .. Omgaga. Im sooo im gunna CRy. I've been at this dentist since 11.. I was suposed 2 just get a crown put on (30mins)... |
|       5|          0| Sentiment140    | i think mi bf is cheating on me!!! T\_T                                                                                   |
|       6|          0| Sentiment140    | or i just worry too much?                                                                                                 |

``` r
prop.table(table(raw_tweets[, "Sentiment"]))
```

    ## 
    ##         0         1 
    ## 0.4994473 0.5005527

``` r
prop.table(table(raw_tweets[, "SentimentSource"]))
```

    ## 
    ##       Kaggle Sentiment140 
    ##  0.000845051  0.999154949
