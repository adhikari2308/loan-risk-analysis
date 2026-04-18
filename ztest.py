import pandas as pd
import numpy as np
from scipy.stats import norm

df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\cs2 python\loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

sample_mean = df['income_annum'].mean()
population_mean = 5000000   # assumed population mean
sample_std = df['income_annum'].std()
n = len(df)

z_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
p_val = 2 * (1 - norm.cdf(abs(z_stat)))

print("===== Z-Test for Annual Income =====")
print("Sample Mean:", sample_mean)
print("Population Mean:", population_mean)
print("Z-Statistic:", z_stat)
print("P-Value:", p_val)

if p_val < 0.05:
    print("Result: Reject Null Hypothesis")
else:
    print("Result: Fail to Reject Null Hypothesis")
