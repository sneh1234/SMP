from http.server import BaseHTTPRequestHandler

import yfinance as yf

class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/plain')

        dl = dataloader()
        dl.get_stock_data('TATASTEEL')

        self.end_headers()
        self.wfile.write('Hello, world2!'.encode('utf-8'))

        self.wfile.write("<img src='../plotsmy_plot.png'/>")
        return



class dataloader():
    def __init__():
        passs
    def get_stock_data(stock):
        ticker = yf.Ticker('{}.NS'.format(stock))
        aapl_df = ticker.history(period="5y")
        fig, ax = plt.subplots()
        s.plot.bar()
        fig.savefig('../plots/my_plot.png')



