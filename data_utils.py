import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


def preprocess_data(path):
  """
  Parse WhatsApp chat history file to get nb of messages exchanged per day
  """
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  pattern = '^[0-9]{2}/[0-9]{2}/[0-9]{4},$'
  dates = [line[:10] for line in lines if re.match(pattern, line[:11])]
  counter = Counter(dates)
  dates, counts = zip(*counter.items())
  data = pd.DataFrame({'date': dates, 'count': counts})
  data['date'] = pd.to_datetime(data.date, format='%d/%m/%Y').dt.date
  data = data.sort_values(by='date')
  data = data.set_index(data.date)
  data = data[['count']]
  return data


def load_data(path):
  return pd.read_csv(path)


def plot_data(data):
  dates = pd.to_datetime(data['date']).dt.date.values
  counts = data['count'].values
  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(111)
  ax.bar(dates, counts)
  ax.set_title('Number of messages exchanged per day')
  ax.set_xlabel('Date')
  ax.set_ylabel('Count')
  return fig
