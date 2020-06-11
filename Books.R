# load libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

# This dataset has been downloaded from 
# https://www.kaggle.com/jealousleopard/goodreadsbooks
# A few records had special characters in the title (, and ") 
# and were throwing parsing errors.
# The errors have been corrected in the local copy of the CSV file

# read the CSV file
dataset <- read_delim("books.csv", delim = ",")

# This returns the structure of the file which will show 11,127 records
str(dataset)

# Examine the dataset to see if any records have 0 average ratings
dataset %>% filter(average_rating == 0) %>% summarize(n = n())

# 26 records have a zero average rating, we will exclude this from the data set
dataset <- dataset %>% filter(average_rating != 0)

# checking row count to see if we now have 11,101 records
nrow(dataset)

# Now splitting the data set into 3 sections
# Based on various articles researched online, 
# a common split for training/testing/validation is 70/15/15
# We will first consider 85% for the training & testing partition set 
#and 15% for the validation set
set.seed(1983, sample.kind = "Rounding")
test_index_1 <- createDataPartition(y = dataset$average_rating, times = 1,
                                  p = 0.85, list = FALSE)
partition <- dataset[test_index_1,]
validation <- dataset[-test_index_1,]

# Now splitting partition into a training set and a testing set
test_index_2 <- createDataPartition(y = partition$average_rating, times = 1,
                                  p = 70/85, list = FALSE)
train_set <- partition[test_index_2,]
test_set <- partition[-test_index_2,]

#checking to see if the number of rows is what we expected
nrow(train_set)
nrow(test_set)
nrow(validation)

# checking correlation between number of pages and book rating
cor(train_set$average_rating, train_set$num_pages)

# checking correlation between number of ratings and book rating
cor(train_set$average_rating, train_set$ratings_count)

# checking correlation between number of reviews and book rating
cor(train_set$average_rating, train_set$text_reviews_count)

# adding a new column to calculate review rate (% of people who posted a review)
# out of the people who rated it
temp <- train_set %>% 
  mutate(review_rate = ifelse(ratings_count == 0, 0, 
                              text_reviews_count / ratings_count))

# checking correlation between review rate and book rating
cor(train_set$average_rating, temp$review_rate)

# adding a new column to calculate length of the book's title
temp <- temp %>% mutate(title_length = nchar(title))

# checking correlation between length of the book's title and book rating
cor(train_set$average_rating, temp$title_length)

# store overall average of book ratings in the training set
overall_avg = mean(train_set$average_rating)

# define function to format any data set so that we can apply it
# to all data sets. Note that none of these transformations are making use
# of the book ratings or data beyond the specific row in the data
format_df = function(df) {
  
  df <- df %>% 
    # add review rate (number of text ratings / number of ratings)
    mutate(review_rate = ifelse(ratings_count == 0, 0, 
                                text_reviews_count / ratings_count)) %>%
    # add length of book title
    mutate(title_length = nchar(title)) %>%
    # add grouping for number of pages
    mutate(num_pages_group = round(num_pages/100)) %>%
    # add grouping for review rate
    mutate(review_rate_group = case_when(
      review_rate < 0.15 ~ round(review_rate, 2),
      TRUE ~ 0.16)) %>%
    
    # regroup by language. Languages with low book count grouped together. 
    group_by(language_code) %>%
    mutate(language_group = case_when(
      language_code == "en-US" ~ "eng",
      language_code == "en-GB" ~ "eng",
      n() > 10 ~ language_code, 
      TRUE ~"other")) %>%
    ungroup() %>%
    
    # group title length by category
    mutate(title_length_group = case_when(
      title_length <= 10 ~ "0-10",
      title_length <= 20 ~ "11-20",
      title_length <= 30 ~ "21-30",
      title_length <= 40 ~ "31-40",
      title_length <= 50 ~ "41-50",
      title_length <= 60 ~ "51-60",
      title_length <= 70 ~ "61-70",
      title_length <= 80 ~ "71-80",
      title_length <= 90 ~ "81-90",
      title_length <= 100 ~ "91-100",
      TRUE ~ "101+"
    ))
}

# add data transformations to training set
train_set <- format_df(train_set)
# add data transformations to test set
test_set <- format_df(test_set)

# We will add to the linear prediction model: Review Rate, Number of pages,
# and length of the book's title
fit <- lm(average_rating ~ title_length + review_rate +
            num_pages, data = train_set)
y_hat_lm <- predict(fit, test_set)
# calculate RMSE for linear model prediction
sqrt(mean((y_hat_lm - test_set$average_rating)^2))

# # train model using randomForest
set.seed(1983, sample.kind = "Rounding")
train_rf <- randomForest(average_rating ~ title_length + review_rate +
                           num_pages + authors + language_group, 
                         data = train_set, 
                         ntree = 180, 
                         importance=TRUE)
# # see importance of each variable
varImp(train_rf)
# 
# # predict test set using RandomForest
y_hat_rf <- predict(train_rf, test_set)
# # calculate RMSE
sqrt(mean((y_hat_rf - test_set$average_rating)^2))

# calculate biases within the training set
train_set <- train_set %>%
  
  # calculate effect of language
  group_by(language_group) %>%
  mutate(b_language = mean(average_rating - overall_avg)) %>%
  ungroup() %>%
  
  # calculate effect of author, after deducting language
  group_by(authors) %>%
  mutate(b_authors = mean(average_rating - overall_avg - b_language)) %>%
  ungroup %>%
  
  # calculate effect of number of pages, after deducting language
  # and author effect
  group_by(num_pages_group) %>%
  mutate(b_num_pages = mean(average_rating - overall_avg - b_language - 
                              b_authors)) %>%
  ungroup %>%
  
  # calculate effect of title length after deducting the other effects
  group_by(title_length_group) %>%
  mutate(b_title_length = mean(average_rating - overall_avg - b_language -
                                 b_num_pages - b_authors)) %>%
  ungroup() %>%
  
  # calculate effect of review rate, after deducting the other effects
  group_by(review_rate_group) %>%
  mutate(b_review_rate = mean(average_rating - overall_avg - b_language - 
                                b_num_pages - b_authors - b_title_length)) %>%
  ungroup()


# extracting language effects from the training set
language_averages <- train_set %>%
  group_by(language_group) %>%
  summarize(b_language = mean(b_language))

# extracting author effects from the training set
authors_averages <- train_set %>%
  group_by(authors) %>%
  summarize(b_authors = mean(b_authors))

# extracting title length effects from the training set
title_length_averages <- train_set %>%
  group_by(title_length_group) %>%
  summarize(b_title_length = mean(b_title_length))

# extracting number of pages effect from the training set
num_pages_averages <- train_set %>%
  group_by(num_pages_group) %>%
  summarize(b_num_pages = mean(b_num_pages))

# extracting review rate effects from the training set
review_rate_averages <- train_set %>%
  group_by(review_rate_group) %>%
  summarize(b_review_rate = mean(b_review_rate))


# calculate predicted ratings for the test set
predicted_ratings <- test_set %>% 
  
  # integrate language effect as extracted from training set
  left_join(language_averages, by='language_group') %>% 
  # applying a default value for categories not found
  mutate(b_language_clean = ifelse(is.na(b_language), 0, b_language)) %>% 
  
  # integrate author effect as extracted from training set
  left_join(authors_averages, by='authors') %>% 
  mutate(b_authors_clean = ifelse(is.na(b_authors), 0, b_authors)) %>% 
  
  # integrate number of pages effect as extracted from training set
  left_join(num_pages_averages, by='num_pages_group') %>% 
  mutate(b_num_pages_clean = ifelse(is.na(b_num_pages), 0, 
                                    b_num_pages)) %>% 
  
  # integrate title length effect as extracted from training set
  left_join(title_length_averages, by='title_length_group') %>% 
  mutate(b_title_length_clean = ifelse(is.na(b_title_length), 0, 
                                       b_title_length)) %>% 
  
  # integrate review rate effect as extracted from training set
  left_join(review_rate_averages, by='review_rate_group') %>% 
  mutate(b_review_rate_clean = ifelse(is.na(b_review_rate), 0, 
                                      b_review_rate)) %>% 
  
  # calculate prediction
  mutate(pred = overall_avg + 
           b_language_clean +
           b_authors_clean + 
           b_num_pages_clean +
           b_title_length_clean +
           b_review_rate_clean) %>%
  # limit prediction to 0.5 to 5.0 range to avoid going outside the range
  mutate(pred_capped = ifelse(pred < 0.5, 0.5, ifelse(pred > 5.0, 5.0, pred))) 

# calculate RMSE for test set
sqrt(mean((predicted_ratings$pred_capped - test_set$average_rating)^2))

# number of books based on title length
train_set %>% group_by(title_length_group) %>%
  summarize(books = n()) %>%
  ggplot(aes(x = title_length_group, y = books)) +
  geom_col(fill = "red", alpha=0.5)

# average rating of books based on title length
train_set %>% group_by(title_length_group) %>%
  summarize(avg_rating = mean(average_rating)) %>%
  ggplot(aes(x = title_length_group, y = avg_rating)) +
  geom_col(fill = "red", alpha=0.5)

# number of books based on review rate
train_set %>% group_by(review_rate_group) %>%
  summarize(books = n()) %>%
  ggplot(aes(x = factor(review_rate_group), y = books)) +
  geom_col(fill = "red", alpha=0.5)

# average rating of books based on review rate
train_set %>% group_by(review_rate_group) %>%
  summarize(avg_rating = mean(average_rating)) %>%
  ggplot(aes(x = review_rate_group, y = avg_rating)) +
  # geom_col(fill = "red", alpha=0.5) +
  geom_line()


# calculate predicted ratings for the validation set
predicted_ratings_2 <- format_df(validation) %>% 
  
  # integrate language effect as extracted from training set
  left_join(language_averages, by='language_group') %>% 
  # applying a default value for categories not found
  mutate(b_language_clean = ifelse(is.na(b_language), 0, b_language)) %>% 
  
  # integrate author effect as extracted from training set
  left_join(authors_averages, by='authors') %>% 
  mutate(b_authors_clean = ifelse(is.na(b_authors), 0, b_authors)) %>% 
  
  # integrate number of pages effect as extracted from training set
  left_join(num_pages_averages, by='num_pages_group') %>% 
  mutate(b_num_pages_clean = ifelse(is.na(b_num_pages), 0, 
                                    b_num_pages)) %>% 
  
  # integrate title length effect as extracted from training set
  left_join(title_length_averages, by='title_length_group') %>% 
  mutate(b_title_length_clean = ifelse(is.na(b_title_length), 0, 
                                       b_title_length)) %>% 
  
  # integrate review rate effect as extracted from training set
  left_join(review_rate_averages, by='review_rate_group') %>% 
  mutate(b_review_rate_clean = ifelse(is.na(b_review_rate), 0, 
                                      b_review_rate)) %>% 
  
  # calculate prediction
  mutate(pred = overall_avg + 
           b_language_clean +
           b_authors_clean + 
           b_num_pages_clean +
           b_title_length_clean +
           b_review_rate_clean) %>%
  # limit prediction to 0.5 to 5.0 range to avoid going outside the range
  mutate(pred_capped = ifelse(pred < 0.5, 0.5, ifelse(pred > 5.0, 5.0, pred))) 

# calculate RMSE for test set
sqrt(mean((predicted_ratings_2$pred_capped - validation$average_rating)^2))



