import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf



ticker = yf.Ticker('RELIANCE.NS')
aapl_df = ticker.history(period="5y")


plt.plot(aapl_df['Close'].values, color = 'black')
plt.savefig('plot.png')
