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
sim_list <- str_replace_all(train$SentimentText, "@[A-z+0-9]*", "at_reference") %>% # Remove at references
        str_replace_all("#[A-z+0-9]*", "hash_tag") %>% # Remove hashtags
        str_replace_all("http:/*[A-z+/+.+0-9]*", "web_link") %>% # Remove web addresses
        str_remove_all("&[A-z+0-9]+;") %>% # Remove &amp; 
        # str_remove_all("[A-C+E-z+0-9+!+.+?+~-ÿ]+") %>%  # Remove all alphanumeric
        tokens(remove_separators = TRUE, what = "word", remove_punct = TRUE, remove_numbers = TRUE) %>% 
        tokens_select(stopwords(), selection = "remove") %>% 
        tokens_wordstem(language = "english") %>% 
        dfm()

sim_list %>% textplot_wordcloud(max_words = 300)
        
str_count()        
        
str_count(raw_tweets$SentimentText[200], "@[A-z+0-9]*")

raw_tweets$web_count <- str_count(raw_tweets$SentimentText, "http:/*[A-z+/+.+0-9]*")
raw_tweets$hashtag_count <- str_count(raw_tweets$SentimentText, "#[A-z+0-9]*")
raw_tweets$at_ref_count <- str_count(raw_tweets$SentimentText, "@[A-z+0-9]*")
raw_tweets$text_length <- nchar(raw_tweets$SentimentText)


raw_tweets$ref_count <- str_count(raw_tweets$SentimentText, "#[A-z+0-9]*") + str_count(raw_tweets$SentimentText, "@[A-z+0-9]*")


ggplot(raw_tweets, aes(x = text_length, fill = Sentiment)) +
        geom_histogram(binwidth = 5) +
        labs(y = "Text Count", x = "Length of Text",
             title = "Distribution of Text Lengths with Class Labels")

library(textstem)
library(quanteda)

set.kRp.env(TT.cmd = "manual", lang = "en", 
            path = "~/Users/Kevin/Documents/TreeTagger/", 
            preset = "eng", 
            validate = TRUE)
set.kRp.env(TT.cmd = "manual", lang = "en", 
            path = "~/Users/Kevin/Documents/TreeTagger/bin", 
            preset = "eng", 
            validate = TRUE)
set.kRp.env(TT.cmd = "manual", lang = "en", 
            path = "~/Users/Kevin/Documents/TreeTagger/cmd", 
            preset = "eng", 
            validate = TRUE)


train_tokens <- tokens(train$SentimentText, what = "word",
                       remove_numbers = TRUE, remove_punct = TRUE, remove_twitter = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)


lemmatize_strings(head(train_tokens), dictionary = lemma_dictionary)
lemma_dictionary <- make_lemma_dictionary(head(train_tokens), engine = 'hunspell')
lemma_dictionary
lemma_dictionary <- make_lemma_dictionary(head(train_tokens, n=50), engine = 'hunspell')
lemma_dictionary <- make_lemma_dictionary(head(train_tokens, n=3), engine = 'hunspell')
lemma_dictionary <- make_lemma_dictionary(train_tokens[[5]], 
                                          engine = 'treetagger', 
                                          path = "/Users/Kevin/Documents/TreeTagger")
lemmatize_words(as.vector(head(train_tokens)))

head(train_tokens)

test_lem <- sapply(train_tokens, lemmatize_words)


train_tokens <- tokens_tolower(train_tokens)
train_tokens <- tokens_select(train_tokens, stopwords(), 
                              selection = "remove")
#train_tokens <- sapply(train_tokens, lemmatize_words)
#train_tokens <- tokenize(train_tokens, format = "obj")
# train_tokens <- tokens_select(train_tokens, stopwords(), 
#                              selection = "remove")
treetag(train_tokens, treetagger = "manual", format = "obj",lang = "en", TT.options=list(path="~/Users/Kevin/Documents/TreeTagger/", preset = "en"))
train_dfm <- dfm(train_tokens, tolower = FALSE)
train_dfm %>% textplot_wordcloud()
train_dfm <- as.matrix(train_dfm)

train_lem_df <- cbind("Sentiment" = as.factor(train$Sentiment), as.data.frame(train_dfm))
kable(train_df[1:10, 1:15])

train$SentimentText[3]

write.csv(train_tokens, file = "train_lem.csv")

vec<-paste(unlist(train$SentimentText), sep = " ", collapse = " ")
vec[which(c(1, diff(vec)) != 0)]

vec<-paste(unlist(train$SentimentText), sep = " ", collapse = " ")
write.csv(vec, file = "train_lem.csv")
# cat train_lem.csv | tree-tagger-english > train_lem_final.txt

train_lem_final <- read_tsv("train_lem_final.txt", col_names = c("Word", "POS", "Lema"))
index_lem <- which(train_lem_final$Lema %in% c("<unknown>", NA))
train_lem_final <- train_lem_final[-index_lem, ]

library(dplyr)
train_lem_final <- left_join(train_lem_final, POS_tbl, by = c("POS"="POS.Tag"))

train_lem_final <- unique(train_lem_final)
stop_vec <- train_lem_final$Word %in% stopwords()
train_lem_final$stop <- stop_vec 








