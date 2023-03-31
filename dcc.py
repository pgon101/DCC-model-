# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:19:01 2023

@author: pgonr
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# Cargando datos
data = pd.read_stata('C:/Users/pgonr/Downloads/felipe1.dta')
    
# Definición función lml
def log_likelihood(beta, data):
    alpha = beta[16]
    mu = beta[17]
    s_n = abs(beta[18])
    s_e = abs(beta[19])
    s_ne = np.sqrt(s_n ** 2 + s_e ** 2)
    rho = s_n / s_ne

    rhs = beta[0] + beta[1] * data['pp_'] + beta[2] * data['tem'] + \
          beta[3] * data['media_nt'] + beta[4] * data['terremoto'] + \
          beta[5] * data['com_2'] + beta[6] * data['com_3'] + \
          beta[7] * data['com_4'] + beta[8] * data['com_5'] + \
          beta[9] * data['com_6'] + beta[10] * data['com_7'] + \
          beta[11] * data['com_8'] + beta[12] * data['com_9'] + \
          beta[13] * data['com_10'] + beta[14] * data['com_11'] + \
          beta[15] * data['com_1']

    t1 = ((data['lnw1']) - rhs - alpha *(data['logp1']) - mu * data['y1_1']) / s_n
    t2 = ((data['lnw1']) - rhs - alpha *(data['logp2']) - mu * data['y2_1']) / s_n
    s1 = ((data['logx1']) - rhs - alpha * (data['logp1']) - mu * data['y1_1']) / s_ne
    s2 = ((data['logx1']) - rhs - alpha * (data['logp2']) - mu * data['y2_1']) / s_ne
    s =( (data['logx1']) - rhs - alpha * (data['pp_']) - mu * data['y']) / s_ne
    u1 = ((data['logx1']) -(data['lnw1'])) / s_e
    r1 = (t1 - rho * s1) / np.sqrt(1 - rho ** 2)
    r2 = (t2 - rho * s2) / np.sqrt(1 - rho ** 2)

    tar = data['tariff_str']

    log_likelihood = np.sum(
        np.log(norm.pdf(s1) * (1 - tar) / s_ne) +
        np.log(norm.pdf(s1) * norm.cdf(r1) +
               norm.pdf(s2) * (1 - norm.cdf(r2)) +
               norm.pdf(u1) * (norm.cdf(t2) - norm.cdf(t1))) * tar)

    return -log_likelihood

# Define parametros iniciales para opti
beta0 = np.array([1] * 20)
bounds = [(-np.inf, np.inf)] * 20
bounds[18] = (0, np.inf)
bounds[19] = (0, np.inf)

# Find the maximum likelihood estimates
result = minimize(log_likelihood, beta0, args=(data,), bounds=bounds)

# Print results
print(result)
print("Estimated parameters:", result.x)
print("Log-likelihood:", -result.fun)