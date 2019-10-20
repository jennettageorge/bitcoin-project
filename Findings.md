# Bitcoin Project

The goal of this project was to do an exploratory analysis of four different bitcoins, BTC, DOGE, ETH, and XRP, from the bitcoin.io API.

Due to the open ended nature of this project and the constraint on time, I chose to focus my analysis on running a multivariate LSTM model on the four time series in the hopes of predicting bitcoin prices.

LSTM was a great choice for this problem because, unlike classic linear regression models, it handles multivariate timeseries data very well. I used the RMSE score of testing and training data as a main indicator of success in the models. 

### How and why I chose the model and parameters.

Long-short term memory (LSTM) is a type of Recurrent Neural Network (RNN) that is best suited for classifying, processing and predicting time series data, especially multivariate time series data like we have. The beauty in an LSTM model over other RNNs is that they solve the vanishing gradients problem that occurs when back propogating, since LSTM units allow gradients to flow through the network unchanged (the error remains in the LSTM unit's cell). The reason I chose this model over a linear regression model like ARIMA, VAR, or VARMA is because after testing my data on all 4 models, I quickly found that LSTM far outperformed the other models in a much shorter amount of time. The parameters of interest in this model were epochs, lookback, and units of RNN (ofcourse there are other hyperparameters involved, but these were the ones I focused on tuning). Based off of prior knowledge, I came into testing with epoch = 50, lookback = 1 and unit = 1. I had fairly good results, but wanted to see if I could do better. No matter how much I increased my parameters, the payoff was not enough to warrant the added computation time, so I decided to stick to these for now.


### Findings 

The trend seems to be more of less the same across all bitcoins. They all started out the year strong and declined towards the end of the year quite rapidly, for all except for DOGE Coin. My knowledge of the subject matter would tell me that this was probably due to a cultlike popularity with the DOGE meme itself which grew in 2018, and not so much with the economics of bitcoin. All four datasets are skewed right as seen in the boxplots, confirming that prices started to drop off earlier in the year.


### Next Steps

You will notice that there is no feature extraction included in this project, since I only had a small number of parameters at my disposal from the API. However there are many more factors that could be considered (i.e. stock market / other financial time series and other events that have causal links to bitcoin price fluctuation). We could also look at a much longer time series with many more data points, and we can choose different forms of data aggregation. These will all effect how our time series model behaves and could have positive or negative impacts on it.
