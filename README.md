# vol_prediction
This project will attempt to predict changes in the implied volatility (IV) of stocks, commodities and indexes over a given period. In the stock market, volatility measures how much the market moved relative to its expectation. Essentially, the volatility can be calculated using the standard deviation of a given period of stock prices. This calculation will yield the “realized volatility” or historical volatility. On the other hand, IVis a measure of market risk and investor sentiment, and represents the expected volatility for the next 30-days. 

## Introduction
The IV is typically predicted based on the price of options 30 days from expiration. For example, the IV of the S\&P 500 is calculated using the prices of options on the SPX index. Options become more valuable when the market is expected to be more volatile. Therefore, being able to predict changes in volatility can be used as a profitable trading strategy. When volatility is expected to expand, buying options can be profitable, and when volatility is expected to decrease, selling options is expected to be profitable. 

For example, Figure 1 shows a plot of historical volatility and the price of the S\&P 500. As you can see, they are partially inversely correlated. 

![Figure 1](https://github.com/edm54/vol_prediction/blob/main/vol_spy.png)
<img src="https://github.com/edm54/vol_prediction/blob/main/vol_spy.png" width="400" height="200">

This project uses historical data for the IV for the S&P 500 (represented as the price of the VIX) and historical price data. The IV data will be taken from the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/VIXCLS). Alternatively a volitility ETF can be used with the pyEX API (such as VXX). 

This website also has values of IV for some stocks like Apple and Amazon, and commodities such as crude oil and gold. However, there is historical data from 1990 to current day, so there is plenty of data for the S&P 500. However, if it is not enough to train the model, I may resort to using some of the other data too. There are several APIs available for getting historical stock data in Python that I will use to get historical S&P 500 price data.

## Project Goal
I will look to predict the change in the IV or VIX using two weeks worth of data on the value of the S\&P 500. In general, if the market drops, the VIX is expected to increase, but how it moves is also relevant. If the market falls rapidly, the VIX will rise much more than if the market falls slowly. This project can also have numerous extensions, such as attempting to predict volatility expansion by using a set period of data to predict the IV 2 weeks later. This could be used to hedge a portfolio, since IV rises when the market falls. So, theoretically, this model could be used to predict market corrections. 

Since the input to the model will be a series of price data, my first approach will be to use an RNN, likely an LSTM model. Depending on the success of the RNN, I may look to use a CNN with 1-dimensional convolution operations to see how this approach performs. 

Note that while predicting stock movement is quite difficult as it is almost entirely random, volatility is known to be mean reverting and therefore may be able to be modeled. One difficulty I see is that change in IV also depends on the value of IV at the beginning of the period. IV will fall faster when it is at highly elevated levels. I will likely try to find a way to work that into the model. One approach would be to use an Seq2Seq model when the output sequence is the sequence of IV values, and the input to the decoder could be the starting value of IV. 

## How to use this framework
To run the code, you must first download volitility data from [here](https://fred.stlouisfed.org/series/VIXCLS). I have also pushed a data file for the SPY to avoid having to download any data. The FRED has a several volitility data sets for stocks (AAPL, AMZN, IBM), indexes (Nasdaq, S&P 500, Russel) or even commodities (gold and oil). 
The code can be run with the following arguments: `-l -s SPY -t`

For some reason I had trouble with saving and loading the model when using Ubuntu and running the code as a module from the command line.

This command will load `-l` the SPY `-s` data file and will train `-t` an LSTM to predict the IV change over 10 days. 
Other stocks can be used with the `-s` flag, but you must input the path to the volatility file using the `-v` flag. 
The `-c` flag will use the combined model, which inputs the initial vol into the network in a concatenation layer after the LSTM. Note this approach did not actually perform better than the standard LSTM.


## Results
The LSTM model did much better on the SPY data than the data for Apple or Amazon. 
Below are the results for the SPY (S&P 500). The predicted IV has a .96 correlation to the actual IV, which is quite good. This is because the IV change is often small, so most of the time the error in IV is small (even if the error in IV change is large) 

![Figure 2]( https://github.com/edm54/vol_prediction/blob/main/Predicted_vol_orig.png)
To show the results in better detail, I have zoomed in, shown below.
![Figure 3](https://github.com/edm54/vol_prediction/blob/main/Predicted_vol_orig_400_600.png)

As you can see, not too bad. However, these plots do not explictly show what the network is predicting, but rather show the predicted IV change plus the initial IV, which makes the results look better than they are. 
Below I have shown the results of the network, comparing the predicted change vs the actual change. As you can see, the network still does a decent job, but tends to underestimate the large moves. 
![Figure 4]( https://github.com/edm54/vol_prediction/blob/main/delta%20v.png)

I have zoomed in on the same region as above, which is not too bad. In fact, the predictions are correlated to the targets with a correlation coefficient of .65. 
![Figure 5]( https://github.com/edm54/vol_prediction/blob/main/delta_400_600.png)

