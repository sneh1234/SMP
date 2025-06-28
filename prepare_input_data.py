

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yfinance as yf


c = 0
for stock in ['PRAJIND', 'DABUR', 'BATAINDIA', 'BAJFINANCE', 'ADANIENT',
       'HAVELLS', 'FSL', 'EXIDEIND', 'TCS', 'PIDILITIND', 'UBL',
       'COFORGE', 'MPHASIS', 'BERGEPAINT', 'BAJAJFINSV', 'MUTHOOTFIN',
       'ABFRL', 'RELIANCE', 'BRITANNIA', 'JUSTDIAL', 'RBLBANK', 'VBL',
       'HINDUNILVR', 'AXISBANK', 'AUROPHARMA', 'DIVISLAB', 'ASIANPAINT',
       'TATAELXSI', 'AARTIIND', 'ADANIPOWER', 'GLENMARK', 'JSWSTEEL',
       'KOTAKBANK', 'IPCALAB', 'LTTS', 'GAIL', 'LTIM', 'TECHM',
       'GODREJCP', 'NAUKRI', 'MARICO', 'JUBLFOOD', 'DRREDDY',
       'NATIONALUM', 'ASTRAL', 'TATACHEM', 'CUB', 'SBIN', 'BSOFT',
       'INDHOTEL', 'TRENT', 'LALPATHLAB', 'SRF', 'TITAN', 'LT', 'HSCL',
       'HINDALCO', 'PEL', 'DMART', 'CONCOR', 'MCDOWELL-N', 'INDIANB',
       'LUPIN', 'BIOCON', 'BOSCHLTD', 'SUNPHARMA', 'CHAMBLFERT', 'WIPRO',
       'MRF', 'INDUSINDBK', 'CIPLA', 'CHOLAFIN', 'APOLLOTYRE', 'GNFC',
       'POWERGRID', 'CANBK', 'GMRINFRA', 'ZYDUSLIFE', 'ULTRACEMCO',
       'TORNTPOWER', 'LICHSGFIN', 'SAIL', 'ACC', 'SHRIRAMFIN',
       'EICHERMOT', 'PETRONET', 'MARUTI', 'PIIND', 'RIIL', 'APOLLOHOSP',
       'UJJIVAN', 'UPL', 'HFCL', 'MANAPPURAM', 'NTPC', 'BAJAJ-AUTO',
       'ONGC', 'CUMMINSIND', 'TATAPOWER', 'AMBUJACEM', 'TATACONSUM',
       'ABB', 'PERSISTENT', 'FORTIS', 'GODREJPROP', 'HEROMOTOCO',
       'COALINDIA', 'SIEMENS', 'BANKINDIA', 'NMDC', 'FEDERALBNK',
       'TATACOMM', 'BANKBARODA', 'IDFC', 'ESCORTS', 'CROMPTON', 'YESBANK',
       'ZEEL', 'HINDCOPPER', 'DLF', 'TATASTEEL', 'MOTHERSON',
       'IBULHSGFIN', 'POONAWALLA', 'RELINFRA', 'JINDALSTEL', 'BHARTIARTL',
       'GUJGASLTD', 'IDBI', 'SUNTV', 'IGL', 'JSWENERGY', 'VOLTAS',
       'ALKEM', 'BHEL', 'LAURUSLABS', 'IRB', 'TVSMOTOR', 'RPOWER', 'MGL',
       'RCF', 'RECLTD', 'NAVINFLUOR', 'IOB', 'BALRAMCHIN', 'IOC',
       'GRASIM', 'IDFCFIRSTB', 'WOCKPHARMA', 'RENUKA', 'CEATLTD', 'BPCL',
       'DELTACORP', 'UNIONBANK', 'PNB', 'APLAPOLLO', 'PFC', 'INDIGO',
       'NCC', 'INDIACEM', 'NBCC', 'BHARATFORG', 'CENTURYTEX', 'CGPOWER',
       'VEDL', 'TATAMOTORS', 'STAR', 'SUZLON', 'TV18BRDCST', 'HINDPETRO',
       'BEML', 'CANFINHOME', 'IDEA', 'RAIN', 'PATANJALI', 'IBREALEST',
       'BSE', 'GRAPHITE', 'HEG']:
    print(stock)
    c = c + 1
    print(c)
    ticker = yf.Ticker(stock + '.NS')
    stock_recent_data = ticker.history(period="6y")
    if os.path.exists('Validation_Data/{}.NS.csv'.format(stock)):
        df = pd.read_csv('Validation_Data/' + stock + '.NS.csv')
        stock_recent_data = stock_recent_data.reset_index()
        stock_recent_data['Date'] = stock_recent_data['Date'].apply(lambda x: str(x).split(' ')[0])
        df['Date'] = df['Date'].apply(lambda x: x.split(' ')[0])
        stock_recent_data = df.append(stock_recent_data[~stock_recent_data['Date'].isin(df['Date'].unique())])
    stock_recent_data.to_csv('Validation_Data/{}.NS.csv'.format(stock), index =False)
    
