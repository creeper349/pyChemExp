import numpy as np
from ..devint import *

def test_func(x:float):
    return np.exp(-x)+np.sin(x)-x**2

def test_func_dev(x:float):
    return -np.exp(-x)+np.cos(x)-2*x

print(f"Rectangle:{intergration_func(-2,3,test_func,INTEGRATION_TYPE.RECTANGLE,n_domains=100,absolute=False)}")
print(f"Trapzoid:{intergration_func(-2,3,test_func,INTEGRATION_TYPE.TRAPZOID,n_domains=100,absolute=False)}")
print(f"Boole:{intergration_func(-2,3,test_func,INTEGRATION_TYPE.BOOLE,n_domains=100,absolute=False)}")
from scipy.integrate import quad
print(f"scipy:{quad(test_func,-2,3)}")
# real value: -3.75355197605

print(f"Forward:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.FORWARD)}")
print(f"Backward:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.BACKWARD)}")
print(f"Central:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.CENTRAL)}")
print(f"4th-order:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.FOUR_POINTS)}")
print(f"6th-order:{derivative_func(3.0,test_func,DERIVATIZATION_TYPE.SIX_POINTS)}")
# real value: -7.03977956497

x=np.linspace(-2,3,100)
y=test_func(x)
dev=derivative_discrete(x,y)
print(f"relative discrete derivative error={((dev-test_func_dev(x))/test_func_dev(x)).mean()}")
print(f"integration of array:{integration_discrete(x,y,absolute=False)}")