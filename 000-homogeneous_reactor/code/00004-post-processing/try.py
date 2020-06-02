import pandas as pd
import cantera as ct

path = '/media/pascal/TOSHIBA EXT/BA/{}_{}_samples.csv'.format('000', 'train')
data = pd.read_csv(path)

# %% Select only the data needed for the plot
data = data[data.pode == 3]
data = data[data.P_0 == 20 * ct.one_atm]
data = data[data.T_0 == 950]

# %%
data2 = data[data.phi == 1.0]
phi = data[['phi']]

# %%
phi = phi.drop_duplicates()
