import numpy as np
import matplotlib.pyplot as plt
from ..interpolation import *

a=np.linspace(-10,10,40)
b=np.sin(a)+np.cos(2*a)+np.sin(3*a)
type_int=INTERPOLATE_TYPE.CUBIC
fig,ax=plt.subplots()
test_func=Interpolation(a,b,interpo_type=type_int)
test_func.scattering(ax)
test_func.plotting(ax,num_points=200,derivative=False,label="func")
test_func.plotting(ax,num_points=200,derivative=True,label="derivative")
print(test_func.integration(-3,2)) # should be -1.7157215
plt.show()