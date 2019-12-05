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

The following is an analysis of the *Twitter Sentiment Analysis Dataset* available at: <http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/>. I will attempt to use this data to train a model to predict the sentiment in future tweets. I will walk through my methodology below and include code. The github repo for my work can be found here: <https://github.com/kwbonds/TwitterSentimentAnalysis>. The file is &gt; 50 MB, so I have not included it in the repo. You will need to download it from the source above and place it in a file called *data* (see code below).

Load Data from .zip file
------------------------

``` r
raw_tweets <-  read_csv(unzip("data/Sentiment-Analysis-Dataset.zip"))
```

The Data
--------

Take a quick look at what we have.

``` r
str(raw_tweets)
```

    ## Classes 'tbl_df', 'tbl' and 'data.frame':    1578603 obs. of  4 variables:
    ##  $ ItemID         : num  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ Sentiment      : num  0 0 1 0 0 0 1 0 1 1 ...
    ##  $ SentimentSource: chr  "Sentiment140" "Sentiment140" "Sentiment140" "Sentiment140" ...
    ##  $ SentimentText  : chr  "is so sad for my APL friend............." "I missed the New Moon trailer..." "omg its already 7:30 :O" ".. Omgaga. Im sooo  im gunna CRy. I've been at this dentist since 11.. I was suposed 2 just get a crown put on (30mins)..." ...
    ##  - attr(*, "problems")=Classes 'tbl_df', 'tbl' and 'data.frame': 27 obs. of  5 variables:
    ##   ..$ row     : int  4285 4285 4286 4286 4287 4287 4287 4287 4287 4287 ...
    ##   ..$ col     : chr  "SentimentText" "SentimentText" "SentimentText" "SentimentText" ...
    ##   ..$ expected: chr  "delimiter or quote" "delimiter or quote" "delimiter or quote" "delimiter or quote" ...
    ##   ..$ actual  : chr  " " " " " " " " ...
    ##   ..$ file    : chr  "'./Sentiment Analysis Dataset.csv'" "'./Sentiment Analysis Dataset.csv'" "'./Sentiment Analysis Dataset.csv'" "'./Sentiment Analysis Dataset.csv'" ...
    ##  - attr(*, "spec")=
    ##   .. cols(
    ##   ..   ItemID = col_double(),
    ##   ..   Sentiment = col_double(),
    ##   ..   SentimentSource = col_character(),
    ##   ..   SentimentText = col_character()
    ##   .. )

|     ItemID|    Sentiment| SentimentSource    | SentimentText                                                                                                                                                                                                                                                                                                                                         |
|----------:|------------:|:-------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|          1|            0| Sentiment140       | is so sad for my APL friend.............                                                                                                                                                                                                                                                                                                              |
|          2|            0| Sentiment140       | I missed the New Moon trailer...                                                                                                                                                                                                                                                                                                                      |
|          3|            1| Sentiment140       | omg its already 7:30 :O                                                                                                                                                                                                                                                                                                                               |
|          4|            0| Sentiment140       | .. Omgaga. Im sooo im gunna CRy. I've been at this dentist since 11.. I was suposed 2 just get a crown put on (30mins)...                                                                                                                                                                                                                             |
|          5|            0| Sentiment140       | i think mi bf is cheating on me!!! T\_T                                                                                                                                                                                                                                                                                                               |
|          6|            0| Sentiment140       | or i just worry too much?                                                                                                                                                                                                                                                                                                                             |
|  We have g|  reater that| 1.5M rows. Even th | ough tweets are somewhat short, this is a lot of data and tokenization will undoubtedly create many more features than can be handled efficiently. We should probably train on about 5% of this data and use as much of the rest as we want to test. we will ofcouse make sure to maintain the proportionality along the way. Let's see what that is. |

What proportion of "Sentiment" do we have in our corpus?

``` r
prop.table(table(raw_tweets[, "Sentiment"]))
```

    ## 
    ##         0         1 
    ## 0.4994473 0.5005527

Looks like it is almost 50/50. Nice. Even though (in this case) a random sample would probably give us very similar proportions, we will use techniques to hard maintain this balance along the way. i.e. just as if we had an unbalanced dataset.

``` r
prop.table(table(raw_tweets[, "SentimentSource"]))
```

    ## 
    ##       Kaggle Sentiment140 
    ##  0.000845051  0.999154949

I'm not sure what this *SentimentSource* column is, but it looks like the vast majority is "Sentiment140". I think we'll ignore it for now.
