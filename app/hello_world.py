print("hello world! let's do some ml ops!")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

values = np.random.randn(100) # array of normally distributed random numbers
s = pd.Series(values) # generate a pandas series
s.plot(kind='hist', title='Normally distributed random values') # hist computes distribution
plt.show()   
