import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ..regression import Regressor
from ..regression.RegressUtils import *

x=np.linspace(2,10,30)
y=hyperbl(x,4.5,6.5)

print(curve_fit(hyperbl,x,y))
regress=Regressor(hyperbl,x,y,max_iter=2000,tol=1e-8,bound=(-5,5))
print(regress.fit())
fig, ax=plt.subplots()
regress.plot(ax)
regress.scatter(ax)
plt.show()