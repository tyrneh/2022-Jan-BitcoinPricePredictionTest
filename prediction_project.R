# clean environment
rm(list=ls())

### Bitcoin Price Prediction Test Project
# Henry Tian Jan 2022
# our objective will be to decide whether bitcoin will open at a higher price on day i+1, based on the data of day i. 
# if we predict a higher open price, then we will want to buy bitcoin at the closing price. 

# since we only use day i's data to predict day i+1, we can combine raw data from test and train files and use cross validation to choose optimal strategy



require(dplyr)

# read training and test csv files
getwd()
setwd("C:/Users/Henry Tian/Box Sync/Github/BitcoinPricePredictionTest")
df_train <- read.csv(file = "data/bitcoin_price_Training - bitcoin_price.2013Apr-2017Aug.csv")
df_test <- read.csv(file = "data/bitcoin_price_1week_Test - bitcoin_price_Aug2017_1week.csv")

dim(df_train)
#check for missing values
sum(is.na(df_train))
sum(is.na(df_test))

#combine both test and train files into single dataframe
df = rbind(df_test, df_train)

# create column of next day open price
df['next_open'] = df$Open
df = mutate(df,next_open = lag(Open))
df = na.omit(df)

# create factor column to indicate if next_open is higher than closing price
df['Buy'] = df$next_open > df$Close
df$Buy = as.factor(df$Buy)

# convert volumne and market cap to numeric
df$Volume = as.numeric(gsub(",", "", df$Volume))
df$Market.Cap = as.numeric(gsub(",", "", df$Market.Cap))
df = na.omit(df)
str(df)

#define design matrix and response vector
x = df[,-c(1,8,9)]
y = as.numeric(df[,9])-1 #set y vector as 0,1 numeric 

#define clean dataframe 
df_final = df[,-c(1,8)] #drop date index, and next day price column


############################################
### Method 1: Simple Logistic Regression ###
############################################
set.seed(1)

logit_fit = glm(Buy ~ ., family = binomial, data = df_final)
summary(logit_fit)

#use CV to find test MSE
library(boot)

#10fold CV MSE of logit model
logit_cv = cv.glm(data = df_final, logit_fit, K = 10)
MSE_logit = logit_cv$delta[2]




############################################
### Method 2: LASSO logistic Regression ####
############################################

library(glmnet)

#use CV to choose best lambda value
cv_lasso = cv.glmnet(x=as.matrix(x), y=y, alpha=1, k=10)
plot(cv_lasso)

#choose best lambda
best_lambda <- cv_lasso$lambda.min

#MSE given best lambda
index_best_lambda = match(best_lambda,cv_lasso$lambda) #index of best lambda
MSE_lasso = cv_lasso$cvm[index_best_lambda]



############################################
### Method 3: Boosting ###########
############################################

library(gbm)
set.seed(1)

# simple first boosting attempt
boost_buy = gbm(Buy ~ ., data = df_final, distribution = 'gaussian', n.trees = 5000, interaction.depth = 4, shrinkage = 0.1, cv.folds = 10)
summary(boost_buy)

print(boost_buy)
min(boost_buy$cv.error)

#plot loss function as a result of n trees added to ensemble
gbm.perf(boost_buy, method = "cv")





# to find best boosted model, we need to perform a grid search, modifying all of our tuning parameters: number of trees, interaction depth, and shrinkage rate
# we also introduce stochastic gradient descent by allowing bag.fraction < 1
# create hyperparameter grid
hyper_grid = expand.grid(
  shrinkage = c(.01, .1, 0.3),
  interaction.depth = c(1, 2, 3),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(0.2, 0.6, 1), 
  optimal_trees = 0,               # a place to dump results
  min_MSE = 0                     # a place to dump results
)

nrow(hyper_grid)



# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = Buy ~ .,
    distribution = "gaussian",
    data = df_final,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    cv.folds = 5,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min cv error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$cv.error)
  hyper_grid$min_MSE[i] <- (min(gbm.tune$cv.error))
}

hyper_grid %>% 
  dplyr::arrange(min_MSE) %>%
  head(10)


# we see that optimal paramets are somewhere between:
# shrinkage from 0.01 to 0.1
# depth from 1 to 2
# min obs from 10 - 15
# bag fraction around 0.2
# optimal trees are all much less than 5000

# let's do another grid search at zoomed in coordinates

# modify hyperparameter grid
hyper_grid_1 <- expand.grid(
  shrinkage = c(.01, .05, .1),
  interaction.depth = c(1,2),
  n.minobsinnode = c(5,10,15),
  bag.fraction = c( 0.05, 0.1, 0.2), 
  optimal_trees = 0,               # a place to dump results
  min_MSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid_1)


# grid search again
for(i in 1:nrow(hyper_grid_1)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = Buy ~ .,
    distribution = "gaussian",
    data = df_final,
    n.trees = 1000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    cv.folds = 5,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min cv error and trees to grid
  hyper_grid_1$optimal_trees[i] <- which.min(gbm.tune$cv.error)
  hyper_grid_1$min_MSE[i] <- (min(gbm.tune$cv.error))
}

hyper_grid_1 %>% 
  dplyr::arrange(min_MSE) %>%
  head(10)





#let's record the optimal model
#    shrinkage interaction.depth n.minobsinnode bag.fraction optimal_trees   min_MSE
#      0.05                 1             15         0.05            12     0.2416318

best_boost = gbm(Buy ~ .
                 , distribution = "gaussian"
                 , data = df_final
                 , n.trees = 1000
                 ,interaction.depth = 1
                 ,shrinkage = 0.05
                 ,n.minobsinnode = 15
                 ,bag.fraction = 0.05
                 ,cv.folds = 10
                 )

summary(best_boost)
print(best_boost)


#MSE of optimal Boosting method
MSE_boost = min(best_boost$cv.error)

#plot loss function as a result of n trees added to ensemble
gbm.perf(best_boost, method = "cv")






############################################
### Compare MSE of 3 approaches ###########
############################################
CV_MSE = data.frame(Model = c("logit","lasso","boost"), CV_MSE = c(MSE_logit,MSE_lasso,MSE_boost))
CV_MSE

