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
library(doSNOW)
```

The following is an analysis of the *Twitter Sentiment Analysis Dataset* available at: <http://thinknook.com/twitter-sentiment-analysis-training-corpus-data> set-2012-09-22/. I will attempt to use this data to train a model to predict the sentiment in future tweets. I will walk through my methodology below and include code. The github repo for my work can be found here: <https://github.com/kwbonds/TwitterSentimentAnalysis>. The file is &gt; 50 MB, so I have not included it in the repo. You will need to download it from the source above and place it in a file called *data* (see code below).

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

|     ItemID|    Sentiment| SentimentSource    | SentimentText                                                                                                                                                                                                                                                                                                                                                                      |
|----------:|------------:|:-------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|          1|            0| Sentiment140       | is so sad for my APL friend.............                                                                                                                                                                                                                                                                                                                                           |
|          2|            0| Sentiment140       | I missed the New Moon trailer...                                                                                                                                                                                                                                                                                                                                                   |
|          3|            1| Sentiment140       | omg its already 7:30 :O                                                                                                                                                                                                                                                                                                                                                            |
|          4|            0| Sentiment140       | .. Omgaga. Im sooo im gunna CRy. I've been at this dentist since 11.. I was suposed 2 just get a crown put on (30mins)...                                                                                                                                                                                                                                                          |
|          5|            0| Sentiment140       | i think mi bf is cheating on me!!! T\_T                                                                                                                                                                                                                                                                                                                                            |
|          6|            0| Sentiment140       | or i just worry too much?                                                                                                                                                                                                                                                                                                                                                          |
|  We have g|  reater that| 1.5M rows. Even th | ough tweets are somewhat short, this is a lot of data. Tokenization will undoubtedly create many more features than can be handled efficiently if we were to try to use this much data. We should probably train on about 5% of this data and use as much of the rest as we want to test. We will make sure to maintain the proportionality along the way. Let's see what that is. |

What proportion of "Sentiment" do we have in our corpus?

``` r
prop.table(table(raw_tweets[, "Sentiment"]))
```

    ## 
    ##         0         1 
    ## 0.4994473 0.5005527

Looks like almost 50/50. Nice. In this case a random sample would probably give us very similar proportions, we will use techniques to hard maintain this proportion i.e. just as if we had an unbalanced data set.

``` r
prop.table(table(raw_tweets[, "SentimentSource"]))
```

    ## 
    ##       Kaggle Sentiment140 
    ##  0.000845051  0.999154949

I'm not sure what this *SentimentSource* column is, but it looks like the vast majority is "Sentiment140". We'll ignore it for now.

Stratified sample
-----------------

Let's create a data partition. First we'll take 4% of the data for training and validation. We'll reserve the indexes so we can further partition later.

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

So, now we have 3789 tweets. Check proportions just to be safe.

``` r
prop.table(table(train$Sentiment))
```

    ## 
    ##         0         1 
    ## 0.4959092 0.5040908

And we have almost exactly the same proportions as our original, much larger, data set.

Tokenization
------------

Let's now tokenize our text data. This is the first step in turning raw text into features. We want the individual words to become features. We'll cleanup some things, engineer some features, and maybe create some combinations of words a little later. There are lots of decisions to be made when doing this sort of text analysis. Do we want our features to contain punctuation, hyphenated words, etc.? Let's try removing some of this to make things simpler.

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

These are the tokens, from the 29th record, of the training data set. i.e. the tweet below.

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

We see some upper case is present. Let's change all to lower to reduce the possible combinations.

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

Next, we need to stem the tokens. Stemming is a method of getting to the word root. This way, we won't have multiple versions of the same root word. We can illustrate below.

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

We now have a matrix--the length of our original data frame--now with 7758 features in the term. That is a lot of features. We are definitely suffering from the "curse of demensionality". We'll need to do some feature reduction at some point.

``` r
dim(train_dfm)
```

    ## [1] 3789 7775

Let's look at the first 6 documents (as rows) and the first 20 features of the term (as columns).

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

Now we have a nice DFM. The columns are the features, and the column-space is the term. The rows are the documents and the row-space are the corpus.

``` r
train_df <- cbind("Sentiment" = as.factor(train$Sentiment), as.data.frame(train_dfm))
kable(train_df[1:10, 1:15])
```

|        | Sentiment |  last|  night|  uva|  basebal|  come|  boy|  can|  got|  sunburnt|  today|  anoth|  day|  paradis|  swim|
|--------|:----------|-----:|------:|----:|--------:|-----:|----:|----:|----:|---------:|------:|------:|----:|--------:|-----:|
| text1  | 0         |     1|      1|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|
| text2  | 0         |     0|      0|    1|        1|     1|    1|    1|    0|         0|      0|      0|    0|        0|     0|
| text3  | 0         |     0|      0|    0|        0|     0|    0|    0|    1|         1|      1|      1|    1|        1|     1|
| text4  | 1         |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    1|        0|     0|
| text5  | 0         |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|
| text6  | 0         |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|
| text7  | 0         |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|
| text8  | 0         |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|
| text9  | 1         |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|
| text10 | 0         |     0|      0|    0|        0|     0|    0|    0|    0|         0|      0|      0|    0|        0|     0|

``` r
# names(train_df) <- make.names(names(train_df))
names(train_df[60:75])
```

    ##  [1] "scarey"                                        
    ##  [2] "startl"                                        
    ##  [3] "wit"                                           
    ##  [4] "experi"                                        
    ##  [5] "#followfriday"                                 
    ##  [6] "@maria"                                        
    ##  [7] "aguilar@alegria21@lettya@formulacyan@peachonic"
    ##  [8] "#fringeto"                                     
    ##  [9] "internetless"                                  
    ## [10] "state"                                         
    ## [11] "internet"                                      
    ## [12] "black"                                         
    ## [13] "inspir"                                        
    ## [14] "http"                                          
    ## [15] "bit.li"                                        
    ## [16] "6fiep"

Unfortunately, R cannot handle some of these tokens as columns in a data frame. The names cannot begin with an integer or a special character for example. We need to fix these. Here is how.

``` r
names(train_df) <- make.names(names(train_df), unique = TRUE)
names(train_df[60:75])
```

    ##  [1] "scarey"                                        
    ##  [2] "startl"                                        
    ##  [3] "wit"                                           
    ##  [4] "experi"                                        
    ##  [5] "X.followfriday"                                
    ##  [6] "X.maria"                                       
    ##  [7] "aguilar.alegria21.lettya.formulacyan.peachonic"
    ##  [8] "X.fringeto"                                    
    ##  [9] "internetless"                                  
    ## [10] "state"                                         
    ## [11] "internet"                                      
    ## [12] "black"                                         
    ## [13] "inspir"                                        
    ## [14] "http"                                          
    ## [15] "bit.li"                                        
    ## [16] "X6fiep"

Setting up for K-fold Cross Validation
--------------------------------------

We will set up a control plan for 30 models. We should be able to use this plan for all our subsequent modeling.

``` r
set.seed(42)

cv_folds <- createMultiFolds(train$Sentiment, k = 10, times = 3)

cv_cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv_folds)
```

Train the First Model
---------------------

Let's train the first model to see what kind of accuracy we have. Let's use a single decision tree algorithm. This algorithm will, however, create 30 \* 7 or 210 models.

``` r
rpart1 <- train(Sentiment ~ ., data = train_df, method = "rpart", 
                    trControl = cv_cntrl, tuneLength = 7)
```

``` r
rpart1
```

    ## CART 
    ## 
    ## 3789 samples
    ## 7758 predictors
    ##    2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 3410, 3410, 3410, 3410, 3410, 3410, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp           Accuracy   Kappa     
    ##   0.002341671  0.6385143  0.27332586
    ##   0.004257584  0.6297197  0.25561962
    ##   0.004523683  0.6283125  0.25270399
    ##   0.013943587  0.5406930  0.10136887
    ##   0.014191946  0.5399894  0.10030259
    ##   0.024481107  0.5098037  0.04890699
    ##   0.036189462  0.5009174  0.03378669
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.002341671.

Outputting the model results we see that we have an almost 64% accuracy already. That isn't bad. Really we want to get to about 90% if we can. This is already better than a coin flip and we haven't even begun. Let's take some steps to improve things.

To be continued...
------------------
