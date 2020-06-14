import pandas as pd

samples_normal = pd.read_csv('000_train_samples.csv')
delays_normal = pd.read_csv('000_train_delays.csv')

print(samples_normal.shape)

print(samples_normal.iloc[len(samples_normal)-10000000, :])
