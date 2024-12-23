import pandas as pd


df = None

for i in range(1, 6):
    path = f'bars_{i}.csv'
    temp = pd.read_csv(path)
    if df is None:
        df = temp
    else:
        df = pd.concat([df, temp])

df = df.iloc[:, 1:]
df = df.drop_duplicates()

print(df.shape)

df.to_csv('table.csv')
