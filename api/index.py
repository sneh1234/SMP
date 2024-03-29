from http.server import BaseHTTPRequestHandler

import yfinance as yf
import pandas as pd
#import matplotlib.pyplot as plt


import os
class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/plain')

        dl = dataloader()
        aapl_df = dl.get_stock_data('TATASTEEL')

        self.end_headers()
        files = os.listdir()
        for f in files:
            self.wfile.write(f.encode('utf-8') + '\n')


        self.wfile.write(aapl_df.head())

        self.wfile.write("<img src='./plots/my_plot.png'/>")
        return



class dataloader():
    def __init__(self):
        pass
    def get_stock_data(self, stock):
        if not os.path.exists('./plots/my_plot.png'):
                
            ticker = yf.Ticker('{}.NS'.format(stock))
            aapl_df = ticker.history(period="5y")
            #plt.plot(aapl_df['Close'])
            #plt.savefig('./plots/my_plot.png')
            #plt.close()
            return aapl_df
        return 1




