import numpy as np
import pymc3 as pm

import data_utils

data = data_utils.load_data('data/chat_counts_per_day.csv')
n = data.shape[0]

model = pm.Model()

with model:
  alpha = 1.0 / n
  lambda_1 = pm.Exponential('lambda_1', alpha)
  lambda_2 = pm.Exponential('lambda_2', alpha)
  tau = pm.DiscreteUniform('tau', lower=0, upper=n - 1)
  lambda_ = pm.math.switch(tau < np.arange(n), lambda_1, lambda_2)
  observation = pm.Poisson('obs', lambda_, observed=data['count'].values)
  trace = pm.sample(1000, tuning=20000)

texts_per_day = np.zeros(n)
for t in range(n):
  ix = t < trace['tau']
  texts_per_day[t] = 1.0 * (
    trace['lambda_1'][ix].sum() + trace['lambda_2'][~ix].sum()
  ) / trace['tau'].shape

fig = data_utils.plot_data(data)
ax = fig.get_axes()[0]
ax.plot(data['date'], texts_per_day, c='red')
