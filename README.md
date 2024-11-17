import numpy as np

import pandas as pd

import statsmodels.api as sm

from statsmodels.stats.anova import anova_lm

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Data

y = np.array([0.07, 0.09, 0.08, 0.16, 0.17, 0.21, 0.49, 0.58, 0.53, 1.22, 1.15, 1.07, 2.84, 2.57, 3.10])

x = np.array([9.0]*3 + [7.0]*3 + [5.0]*3 + [3.0]*3 + [1.0]*3)

# Create a DataFrame

df = pd.DataFrame({'Y': y, 'X': x})

# Fit a linear model

model = sm.OLS(df['Y'], sm.add_constant(df['X'])).fit()

# Obtain fitted values and residuals

fitted_values = model.fittedvalues

residuals = model.resid

# ANOVA table

anova_table = anova_lm(model)

# Size 0.1 test

p_value = anova_table['PR(>F)']['X']

alpha = 0.1

if p_value < alpha:

print(f"The p-value ({p_value}) is less than {alpha}. Reject the null hypothesis.")

else:

print(f"The p-value ({p_value}) is greater than {alpha}. Fail to reject the null hypothesis.")

# Pairwise comparisons with Bonferroni's method

posthoc = pairwise_tukeyhsd(df['Y'], df['X'], alpha=0.05)

print(posthoc)

# Conclusion about the claim

if model.params['X'] < 0:

print("The chemist's claim that the concentration decreases with time is supported by the findings.")

else:

print("The chemist's claim is not supported by the findings.")
