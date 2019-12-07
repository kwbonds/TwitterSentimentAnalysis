library(tidyverse)
library(readr)
library(ggplot2)
library(caret)

raw_tweets <-  read_csv(unzip("data/Sentiment-Analysis-Dataset.zip"))

raw_tweets$Sentiment <- as.factor(raw_tweets$Sentiment)
raw_tweets$SentimentSource <- as.factor(raw_tweets$SentimentSource)

table(raw_tweets[,2:3])
prop.table(table(raw_tweets[, "Sentiment"]))

raw_tweets[4286,]           


start.time <- Sys.time()


# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)


# As our data is non-trivial in size at this point, use a single decision
# tree alogrithm as our first model. We will graduate to using more 
# powerful algorithms later when we perform feature extraction to shrink
# the size of our data.
rpart1 <- train(Sentiment ~ ., data = train_df, method = "rpart", 
                trControl = cv_cntrl, tuneLength = 7)


# Processing is done, stop cluster.
stopCluster(cl)


# Total time of execution on workstation was approximately 4 minutes. 
total.time <- Sys.time() - start.time
total.time


grepl("\\(:", raw_tweets$SentimentText[1:500]) %>% which()
# [1]  42 284

grepl(":/", str_replace_all(raw_tweets$SentimentText[1:500], "[a-zA-Z0-9_]", "")) %>% which()

# [1]  17  33  42  57  58  66  95 109 118 126 127 130 141 143 146 147 150 151 152 168 169 181 182 186 189
# [26] 190 192 197 200 205 210 218 219 226 228 232 284 310 326 327 332 364 381 384 387 388 390 394 396 398
# [51] 399 410 426 437 441 451 456 457 458 463 472 488 489

str_remove(raw_tweets$SentimentText[42], "http:/*[A-z+/+.+0-9]*")

raw_tweets$SentimentText[200]
# [1] "- @EvertB which one? http://bit.ly/10o8LW, http://bit.ly/Nh82S or http://bit.ly/wr8Vz - latter for the US Store only  #iphone #comics"


###########
# web_link
str_remove_all(raw_tweets$SentimentText[200], "http:/*[A-z+/+.+0-9]*")
# [1] "- @EvertB which one? ,  or  - latter for the US Store only  #iphone #comics"
str_replace_all(raw_tweets$SentimentText[200], "http:/*[A-z+/+.+0-9]*", "web_link")
#[1] "- @EvertB which one? web_link, web_link or web_link - latter for the US Store only  #iphone #comics"

str_replace_all(raw_tweets$SentimentText[1:200], "http:/*[A-z+/+.+0-9]*", "web_link")
###########

###########
# Hashtags
str_remove_all(raw_tweets$SentimentText[200], "#[A-z+0-9]*")
# [1] "- @EvertB which one? http://bit.ly/10o8LW, http://bit.ly/Nh82S or http://bit.ly/wr8Vz - latter for the US Store only   "
str_replace_all(raw_tweets$SentimentText[200], "#[A-z+0-9]*", "hash_tag")
# [1] "- @EvertB which one? http://bit.ly/10o8LW, http://bit.ly/Nh82S or http://bit.ly/wr8Vz - latter for the US Store only  hash_tag hash_tag"

str_replace_all(raw_tweets$SentimentText[1:200], "#[A-z+0-9]*", "hash_tag")
###########

###########
# @ references
str_remove_all(raw_tweets$SentimentText[200], "@[A-z+0-9]*")
# [1] "-  which one? http://bit.ly/10o8LW, http://bit.ly/Nh82S or http://bit.ly/wr8Vz - latter for the US Store only  #iphone #comics"
str_replace_all(raw_tweets$SentimentText[200], "@[A-z+0-9]*", "at_reference")
# [1] "- at_reference which one? http://bit.ly/10o8LW, http://bit.ly/Nh82S or http://bit.ly/wr8Vz - latter for the US Store only  #iphone #comics"

str_replace_all(raw_tweets$SentimentText[1:200], "@[A-z+0-9]*", "at_reference")
###########

str_remove_all(raw_tweets$SentimentText[858], "&[A-z+0-9]+;")

#########
# Find simbals
sim_list <- str_remove_all(raw_tweets$SentimentText[1:3000], "@[A-z+0-9]*") %>% # Remove at references
        str_remove_all("#[A-z+0-9]*") %>% # Remove hashtags
        str_remove_all("http:/*[A-z+/+.+0-9]*") %>% # Remove web addresses
        str_remove_all("&[A-z+0-9]+;") %>% # Remove &amp; 
        # str_remove_all("[A-C+E-z+0-9+!+.+?+~-ÿ]+") %>%  # Remove all alphanumeric
        tokens(remove_separators = TRUE, what = "word") %>% 
        dfm() 
        
        
        

