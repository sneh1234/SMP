import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv('Validation_Data/BAJAJFINSV.NS.csv')
df['Date'] = df['Date'].apply(lambda x: str(x).split(' ')[0])
today = df['Date'].max()

total_days = df[df['Date'] >= '2023-01-01'].shape[0]


df_pred = pd.read_csv('stock_predictions.csv')
df_pred = df_pred[['stock', 'preds']]

df_pred['preds'] = df_pred['preds'].apply(lambda x: pow(np.exp(x), 4) - 1)



df_pred.iloc[:10].to_csv('mysite/static/js/recommendation.csv', index = False)


stocks = df_pred[df_pred['preds'] >= 0.12].stock.unique()
if os.path.exists('mysite/static/js/archives.csv'):
    df = pd.read_csv('mysite/static/js/archives.csv')
else:
    df = pd.DataFrame()

historical_stocks = []

if len(df):
    for stock_l in df['stocks'].values:
        for stock in stock_l.split(':'):
            historical_stocks.append(stock)

historical_stocks = list(set(historical_stocks))
stock_to_remove = []
for stock in historical_stocks:
    if df_pred[df_pred['stock'] == stock].preds.values[0] < 0.07:
        stock_to_remove.append(stock)

print(historical_stocks, stock_to_remove, df, today)
if len(df):
    df = df[df['date'] != today]
df = df.append({'date': today, 'stocks': ':'.join(stocks), 'stocks_to_exit': ':'.join(stock_to_remove), 'returns': 0, 'baseline_returns': 0 }, ignore_index = True)
df = df.reset_index(drop = True)
df.to_csv('mysite/static/js/archives.csv', index = False)
df_stock = {}

for f in os.listdir('Validation_Data/'):
    stock = '.'.join(f.split('.')[:-2])
    stock_data = pd.read_csv('Validation_Data/' + f)

    stock_data['Date'] = stock_data['Date'].apply(lambda x: str(x).split(' ')[0])
    stock_days = stock_data[stock_data['Date'] >= '2023-01-01'].shape[0]
    if stock_days > total_days - 5:
        df_stock[stock] = stock_data




returns = []
baseline_returns = []
for index in range(len(df)-1):
    df1 = df.iloc[index]
    stocks = df1['stocks'].split(':')
    cash = 0


    stocks_invested = {}
    for stock in stocks:
        invested_amt = 1/len(stocks)
        price = df_stock[stock][df_stock[stock]['Date'] == df1['date']]['Close'].values[0]
        stocks_invested[stock] = (invested_amt, price)
    baseline = 0
    for stock in df_stock.keys():
        print(stock, df1['date'])
        baseline = baseline + df_stock[stock]['Close'].values[-1]/df_stock[stock][df_stock[stock]['Date'] == df1['date']]['Close'].values[0]
    baseline = baseline/len(df_stock)

    for index2 in range(index + 1, len(df)):
        df2 = df.iloc[index2]
        if df2['stocks_to_exit'] == df2['stocks_to_exit'] and len(df2['stocks_to_exit']):
            stocks_to_exit = df2['stocks_to_exit'].split(':')
        else:
            stocks_to_exit = []
        for stock in stocks_to_exit:
            if stock in stocks_invested.keys():
                curmp = df_stock[stock][df_stock[stock]['Date'] == df2['date']]['Close'].values[0]
                cash = cash + (stocks_invested[stock][0] * curmp * 0.995) / stocks_invested[stock][1]
                del stocks_invested[stock]

        stocks_to_enter = list(set(df2['stocks'].split(':')) - set(stocks_invested.keys()))
        if len(stocks_to_enter) + len(stocks_invested) >= 10 and len(stocks_invested) < 10 and cash > 0:
            stocks_to_enter = stocks_to_enter[:10-len(stocks_invested)]
            for stock in stocks_to_enter:
                invested_amt = cash/len(stocks_to_enter)
                price = df_stock[stock][df_stock[stock]['Date'] == df2['date']]['Close'].values[0]
                stocks_invested[stock] = (invested_amt, price)
    for stock in stocks_invested.keys():
        curmp = df_stock[stock][df_stock[stock]['Date'] == df2['date']]['Close'].values[0]
        cash = cash + (stocks_invested[stock][0] * curmp)/stocks_invested[stock][1]
    returns.append(100 * (cash - 1))
    baseline_returns.append(100 * (baseline - 1))
returns.append(0)
baseline_returns.append(0)
df['returns'] = returns 
df['baseline_returns'] = baseline_returns
df.to_csv('mysite/static/js/archives.csv', index = False)




df['trading_days'] = 1
df['trading_days'] = len(df) - df['trading_days'].cumsum() - 0
plt.plot(df['trading_days'].values, df['returns'].values, label = 'model')
plt.plot(df['trading_days'].values, df['baseline_returns'].values, label = 'baseline')
plt.title('Historical Performance')
plt.xlabel('trading_days')
plt.ylabel('returns')
plt.legend()
plt.savefig('mysite/static/images/performance_comparision.png')

