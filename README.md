# BitcoinPricePredictionTest

Data from https://www.kaggle.com/team-ai/bitcoin-price-prediction/version/1?select=bitcoin_price_1week_Test+-+bitcoin_price_Aug2017_1week.csv

We'll do a simple machine learning exercise, comparing between logit, lasso, and boosting models to predict whether the next day open price will be higher than closing price. We use CV MSE to choose the optimal model. 

We just want to compare between the models, so for laziness we'll stick with using MSE as the comparison rather than misclassification error rate. 
Using boosting, we are able to reduce CV MSE from 0.246 to 0.241. 
