import math
import random
import requests

import pandas as pd


moskow_center = (55.512223, 37.639650)
spb_center = (59.936014, 30.310829)
moskow_radius = 40 / 111 # 60 killometers

bars = []
key = 'XXX'
sample_num = 100
max_page = 5

using_ids = set()

for sample in range(sample_num):
    print(sample)

    alpha = 2 * math.pi * random.random()

    r = moskow_radius * math.sqrt(random.random())

    point_lat = r * math.sin(alpha) + moskow_center[0]
    point_lon = r * math.cos(alpha) + moskow_center[1]

    for page_num in range(1, max_page + 1):
        request_bar = f'https://catalog.api.2gis.com/3.0/items?q=кафе&sort_point={point_lon}%2C{point_lat}&fields=items.point,items.reviews&key={key}&page={page_num}'
        try:
            req = requests.get(request_bar)
        except ex:
            print(ex)
            print(f'Error request page: {page_num}')
            continue
        if 'result' in req.json():
            #print(req.json())
            for data in req.json()['result']['items']:
                if 'general_rating' in data['reviews'] and 'general_review_count_with_stars' in data['reviews'] and 'address_name' in data:
                    id = data['id']
                    if id not in using_ids:
                        using_ids.add(id)
                        adress = data['address_name']
                        name = data['name']
                        lat = data['point']['lat']
                        lon = data['point']['lon']
                        rating = data['reviews']['general_rating']
                        review_count = data['reviews']['general_review_count_with_stars']
                        bars.append({'id': id, 'adress': adress, 'name': name, 'lat': lat, 'lon': lon, 'rating': rating, 'review_count': review_count})
                else:
                    print(data)
        else:
            print(req.json())
    if sample % 10 == 0:
        df = pd.DataFrame.from_records(bars)
        df.to_csv('bars_5.csv')
