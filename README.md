# vol_prediction
This project will attempt to predict changes in the implied volatility (IV) of stocks, commodities and indexes over a given period. In the stock market, volatility measures how much the market moved relative to its expectation. Essentially, the volatility can be calculated using the standard deviation of a given period of stock prices. This calculation will yield the “realized volatility” or historical volatility. On the other hand, IVis a measure of market risk and investor sentiment, and represents the expected volatility for the next 30-days. 

## Introduction
The IV is typically predicted based on the price of options 30 days from expiration. For example, the IV of the S\&P 500 is calculated using the prices of options on the SPX index. Options become more valuable when the market is expected to be more volatile. Therefore, being able to predict changes in volatility can be used as a profitable trading strategy. When volatility is expected to expand, buying options can be profitable, and when volatility is expected to decrease, selling options is expected to be profitable. 

For example, Figure 1 shows a plot of historical volatility and the price of the S\&P 500. As you can see, they are partially inversely correlated. 

![Figure 1](http://www.cboe.com/publish/micrositecharts/VIX_SP500_Index.jpg)
Figure 1

This project uses historical data for the IV for the S&P 500 (represented as the price of the VIX) and historical price data. The IV data will be taken from the Federal Reserve Economic Data (FRED) [website] (https://fred.stlouisfed.org/series/VIXCLS). Alternatively a volitility ETF can be used with the pyEX API (such as VXX). 

This website also has values of IV for some stocks like Apple and Amazon, and commodities such as crude oil and gold. However, there is historical data from 1990 to current day, so there is plenty of data for the S&P 500. However, if it is not enough to train the model, I may resort to using some of the other data too. There are several APIs available for getting historical stock data in Python that I will use to get historical S&P 500 price data.

## Project Goal
I will look to predict the change in the IV or VIX using two weeks worth of data on the value of the S\&P 500. In general, if the market drops, the VIX is expected to increase, but how it moves is also relevant. If the market falls rapidly, the VIX will rise much more than if the market falls slowly. This project can also have numerous extensions, such as attempting to predict volatility expansion by using a set period of data to predict the IV 2 weeks later. This could be used to hedge a portfolio, since IV rises when the market falls. So, theoretically, this model could be used to predict market corrections. 

Since the input to the model will be a series of price data, my first approach will be to use an RNN, likely an LSTM model. Depending on the success of the RNN, I may look to use a CNN with 1-dimensional convolution operations to see how this approach performs. 

Note that while predicting stock movement is quite difficult as it is almost entirely random, volatility is known to be mean reverting and therefore may be able to be modeled. One difficulty I see is that change in IV also depends on the value of IV at the beginning of the period. IV will fall faster when it is at highly elevated levels. I will likely try to find a way to work that into the model. One approach would be to use an Seq2Seq model when the output sequence is the sequence of IV values, and the input to the decoder could be the starting value of IV. 
