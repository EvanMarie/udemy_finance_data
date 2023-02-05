plot_lines([(msft_full.loc['2016-01-05':'2018-12-31'].close,
			 'close', 'white'),
			(msft_full.loc['2016-01-05':'2018-12-31'].MACD,
			 'MACD', 'cyan'),
			(msft_full.loc['2016-01-05':'2018-12-31'].MACD_Signal,
			 'MACD Signal', 'yellow'),
			(msft_full.loc['2016-01-05':'2018-12-31'].MACD_Hist,
			 'MACD Hist', 'lime')],
		   title = 'Microsoft: Moving Average Convergence Divergence',
		   xlabel = 'Years',
		   ylabel = 'Price',
		  logy = True)

# make the above plots as subplots

