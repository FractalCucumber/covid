import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


data = pd.read_csv("covid_data.csv")

worldwise = data.loc[:, ("Date_reported", "New_cases")].groupby("Date_reported", as_index=False).sum()

x = np.array(worldwise.index)
y = np.array(worldwise["New_cases"])


def f(x, a, b):
    return a*np.exp(b*x)

x = 0.0001*x
p = curve_fit(f, x, y)[0]


print(p)

plt.plot(x, y, x, f(x, *p))
plt.xlabel("day")
plt.ylabel("cases")

plt.show()

