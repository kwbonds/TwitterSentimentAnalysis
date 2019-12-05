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
library(quanteda)
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

Stratified sample
-----------------

Let's create a data partition. First we will take 4% of the data for training and validation. We'll reserve the indexes so that we can further partition later

``` r
set.seed(42)
partition_1_indexes <- createDataPartition(raw_tweets$Sentiment, times = 1, p = 0.004, list = FALSE)
train_validate <- raw_tweets[partition_1_indexes, c(2,4)]
train_indexes <- createDataPartition(train_validate$Sentiment, times = 1, p = 0.60, list = FALSE)
train <- train_validate[train_indexes, ]
test <- train_validate[-train_indexes, ]
nrow(train)
```

    ## [1] 3789

Check proportions just to be safe.

``` r
prop.table(table(train$Sentiment))
```

    ## 
    ##         0         1 
    ## 0.4959092 0.5040908

We can see that we have almost exactly the same proportions.

Tokenization
------------

Let's now tokenize our text data. This is the first step in turning raw text into features. We want to make the individual words as features and do some cleanup. We will also engineer some features and maybe create some combinations of words a little later. The big question is: should we remove numbers, punctuation, hyphens, and symbols?

``` r
train_tokens <- tokens(train$SentimentText, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE, remove_twitter = FALSE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)
```

Let's look at a few to illustrate what we did.

``` r
train_tokens[[29]]
```

    ##  [1] "#millsthemusical" "@lauzzaa"         "i"               
    ##  [4] "hope"             "you've"           "listened"        
    ##  [7] "to"               "some"             "of"              
    ## [10] "these"            "songs"            "they"            
    ## [13] "are"              "so"               "funny"           
    ## [16] "L"                "i"                "still"           
    ## [19] "haven't"          "heard"            "about"           
    ## [22] "ticks"            "x"

These are the tokens from the 29th record from the training data set. i.e. the tweet below.

``` r
train[29,2]
```

    ## # A tibble: 1 x 1
    ##   SentimentText                                                            
    ##   <chr>                                                                    
    ## 1 #millsthemusical  @lauzzaa i hope you've listened to some of these songs…

Also this one:

``` r
train_tokens[[26]]
```

    ##  [1] "quot"         "Salvation"    "has"          "come"        
    ##  [5] "to"           "us"           "He's"         "chosen"      
    ##  [9] "us"           "in"           "love"         "quot"        
    ## [13] "Love"         "it"           "@phatfish"    "@nfellingham"

And we see that we have some upper and lower case. Let's change all to lower.

``` r
train_tokens <- tokens_tolower(train_tokens)
train_tokens[[26]]
```

    ##  [1] "quot"         "salvation"    "has"          "come"        
    ##  [5] "to"           "us"           "he's"         "chosen"      
    ##  [9] "us"           "in"           "love"         "quot"        
    ## [13] "love"         "it"           "@phatfish"    "@nfellingham"

Remove Stopwords
----------------

Let's remove stopwords using the quanteda packages built in *stopwords()* function and look at record 26 again.

``` r
train_tokens <- tokens_select(train_tokens, stopwords(), 
                              selection = "remove")
train_tokens[[26]]
```

    ##  [1] "quot"         "salvation"    "come"         "us"          
    ##  [5] "chosen"       "us"           "love"         "quot"        
    ##  [9] "love"         "@phatfish"    "@nfellingham"

And record 29 again:

``` r
train_tokens[[29]]
```

    ##  [1] "#millsthemusical" "@lauzzaa"         "hope"            
    ##  [4] "listened"         "songs"            "funny"           
    ##  [7] "l"                "still"            "heard"           
    ## [10] "ticks"            "x"

Stemming
--------

Next we can stem the tokens. Stemming is a method of getting to the word root. This way we won't have multiple versions of the same root word. We can illustrate below.

``` r
train_tokens <- tokens_wordstem(train_tokens, language = "english")
train_tokens[[29]]
```

    ##  [1] "#millsthemus" "@lauzzaa"     "hope"         "listen"      
    ##  [5] "song"         "funni"        "l"            "still"       
    ##  [9] "heard"        "tick"         "x"

You can see that "listened" becomes "listen", and "ticks" becomes "tick", etc.

Create a Document-feature Matrix
--------------------------------

``` r
train_dfm <- dfm(train_tokens, tolower = FALSE)
```

``` r
train_dfm <- as.matrix(train_dfm)
```

You can see that we now have a matrix that is the length of our original data frame but now has 7758 features in our term.

``` r
dim(train_dfm)
```

    ## [1] 3789 7775

Let's look at the first 6 documents (and rows) and the first 20 features of the term (as columns).

``` r
kable(head(train_dfm[1:6, 1:20]))
```

|       |  last|  night|  uva|  basebal|  come|  boy|  can|  got|  sunburnt|  today|  anoth|  day|  paradis|  swim|  ship|  pool|  music|  gonna|  danc|  wow|
|-------|-----:|------:|----:|--------:|-----:|----:|----:|----:|---------:|------:|------:|----:|--------:|-----:|-----:|-----:|------:|------:|-----:|----:|
| text1 |     1|      1|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|     0|     0|      0|      0|     0|    0|
| text2 |     0|      0|    1|        1|     1|    1|    1|    0|         0|      0|      0|    0|        0|     0|     0|     0|      0|      0|     0|    0|
| text3 |     0|      0|    0|        0|     0|    0|    0|    1|         1|      1|      1|    1|        1|     1|     1|     1|      0|      0|     0|    0|
| text4 |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    1|        0|     0|     0|     0|      2|      2|     1|    3|
| text5 |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|     0|     0|      0|      0|     0|    0|
| text6 |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|     0|     0|      0|      1|     0|    0|
