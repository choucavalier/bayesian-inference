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
  tau = pm.DiscreteUniform('tau', lower=0, upper=n)
  lambda_ = pm.math.switch(tau < np.arange(n), lambda_1, lambda_2)
  observation = pm.Poisson('obs', lambda_, observed=data['count'].values)
  trace = pm.sample(5000)
