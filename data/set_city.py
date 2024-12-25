import pandas as pd


df = pd.read_csv('/home/nkhamm/study/sna/table.csv')
df['city'] = pd.Series(['' for i in range(df.shape[0])], index=df.index)
df.loc[df['lon'] > 35.0, 'city'] = 'msk'
df.loc[df['lon'] < 32.0, 'city'] = 'spb'
df.to_csv('table.csv')
