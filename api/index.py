from http.server import BaseHTTPRequestHandler

# import yfinance as yf

import os
class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/plain')

        # dl = dataloader()
        # dl.get_stock_data('TATASTEEL')

        self.end_headers()
        self.wfile.write('Hello, world3!'.encode('utf-8'))
        self.wfile.write(os.listdir())

        # self.wfile.write("<img src='../plots/my_plot.png'/>")
        return



# class dataloader():
#     def __init__():
#         pass
#     def get_stock_data(stock):
#         if not os.path.exists('./plots/my_plot.png'):
                
#             ticker = yf.Ticker('{}.NS'.format(stock))
#             aapl_df = ticker.history(period="5y")
#             plt.plot(aapl_df)
#             fig.savefig('../plots/my_plot.png')
#             plt.close()




