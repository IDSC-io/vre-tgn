# %%
# based on: https://towardsdatascience.com/five-confidence-intervals-for-proportions-that-you-should-know-about-7ff5484c024f
import numpy as np
from scipy.stats import norm

def calculate_wald_type_CI(p, n, confidence_level = 0.95):
    standard_dev = np.sqrt(p * (1-p) / n)
    z = norm.ppf(1 - (1 - confidence_level) / 2)  #returns the value of thresholds at which conf.level has to be cut at. for 95% CI, this is -1.96 and +1.96
    ci = [p - z * standard_dev, p + z * standard_dev]
    return ci


# %%

if __name__ == "__main__":
    print(calculate_wald_type_CI(0.81, 5000))
# %%
