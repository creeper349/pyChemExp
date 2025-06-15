import numpy as np
import matplotlib.pyplot as plt
from ..regression.polyregress import PolynomialRegressor, PolyvarRegressor
from ..regression.lasso import LASSO

x=np.linspace(-6,6,25)
y=0.25*x**3+0.45*x**2-1.34*x+9.48

regressor=PolynomialRegressor(x,y,degree=3,lamb=0)
print(regressor.fit())
fig,ax=plt.subplots()
regressor.plot(ax)
regressor.scatter(ax)
plt.show()

x1=np.random.randn(20,5)
x1_=np.concatenate((np.ones((20,1)),x1),axis=1)
w=[1.5,2.3,4.6,7.7,8.9,4.3]
y1=x1_.dot(w)
mulregressor=PolyvarRegressor(x1,y1,lamb=0.0)
print(mulregressor.fit())

y2=np.array([-1.077,-1.308,-1.248,-0.534,0.868,0.721,1.534,-0.392,1.443,-0.205]).reshape(10,1)
X2=np.array([[-1.784, -0.192, 0.505, 0.346],
            [-0.749, 0.762, -1.823, 0.215],
            [-0.636, 0.996, 0.395, 0.747],
            [-0.637, -0.006, -0.051, 0.655],
            [0.455, -0.811, -0.991, -0.268],
            [0.270, -0.633, -0.779, -1.255],
            [0.906, -1.307, -0.524, 2.032],
            [0.844, 1.084, -1.197, -0.255],
            [0.705, -1.287, 0.078, 0.984],
            [-0.637, -0.378, 0.070, 1.140]])

print(LASSO(X2,y2,C=0.1,learning_rate=0.001,max_iter=500))