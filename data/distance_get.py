import requests
import json
import pickle
import time

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('table.csv')
msk = df.loc[df['city'] == 'msk']
data_msk = np.array([[lat, lon] for lat, lon in zip(msk['lat'].iloc, msk['lon'].iloc)])

spb = df.loc[df['city'] == 'spb']
data_spb = np.array([[lat, lon] for lat, lon in zip(spb['lat'].iloc, spb['lon'].iloc)])

cluster_num = 8

clustering_msk = KMeans(n_clusters=8).fit(data_msk)
clustering_spb = KMeans(n_clusters=8).fit(data_spb)

df['cluster_id'] = pd.Series([0 for i in range(df.shape[0])], index=df.index)
df.loc[df['city'] == 'msk', 'cluster_id'] = clustering_msk.labels_
df.loc[df['city'] == 'spb', 'cluster_id'] = clustering_spb.labels_

matrix = np.load('matrix.dat', allow_pickle=True)

print(np.unique(clustering_msk.labels_))

keys = ['2b41609f-e3a1-4ea5-bdba-fa853445a3e4', 'ee16348d-b469-48fb-ad86-6a742f351edb', '0061599d-527e-418b-a5e7-7a8ac8070af6']
keys_ids = 0

url = 'https://routing.api.2gis.com/get_dist_matrix?key={}&version=2.0'
batch_sz = 10
counter = 0


for city in ('msk', 'spb'):
    for i in range(1, cluster_num):
        cur_df = df.loc[(df['city'] == city) & (df['cluster_id'] == i)]
        n = cur_df.shape[0]
        for pos1 in range(0, n, batch_sz):
            for pos2 in range(pos1, n, batch_sz):
                data = {}
                data['points'] = []

                v1, v2 = 0, 0
                for value in cur_df.iloc[pos1:pos1 + batch_sz].iloc:
                    data['points'].append({'lat': value['lat'], 'lon': value['lon']})
                    v1 += 1

                for value in cur_df.iloc[pos2:pos2 + batch_sz].iloc:
                    data['points'].append({'lat': value['lat'], 'lon': value['lon']})
                    v2 += 1

                data['sources'] = [t for t in range(v1)]
                data['targets'] = [t for t in range(v1, v1 + v2)]
                

                val = requests.post(url.format(keys[keys_ids]), json=data)
                val = val.json()
                
                try:
                    for value in val['routes']:
                        matrix[value['source_id'] + pos1, value['target_id'] - v1 + pos1] = [value['distance'], value['duration']]
                        matrix[value['target_id'] - v1 + pos1, value['source_id'] + pos1] = [value['distance'], value['duration']]

                    counter += 1
                except:
                    print(val)
                    keys_ids += 1
                    keys_ids = min(keys_ids, len(keys) - 1)
                    time.sleep(30)

                if counter == 10:
                    counter = 0
                    matrix.dump('matrix.dat')
                    print('waiting...')
                    time.sleep(60)

'''
key = 'a83d4405-baa8-4352-975d-0b955b6899da'
url = 'https://routing.api.2gis.com/async_matrix/create_task/get_dist_matrix?key={}&version=2.0' 

tasks = []

for city in ('msk', 'spb'):
    for i in range(cluster_num):
        cur_df = df.loc[(df['city'] == city) & (df['cluster_id'] == i)]
        data = {}
        data["points"] = [{"lat":val['lat'], "lon":val['lon']} for val in cur_df.iloc]
        data["sources"] = [j for j in range(cur_df.shape[0])]
        data["targets"] = [j for j in range(cur_df.shape[0])]
        val = requests.post(url.format(key), headers={"Content-Type" : "application/json"}, json=data)

        try:
            tasks.append(val.json()['task_id'])
        except:
            print(val.json())

pickle.dump(tasks, 'tasks')
'''