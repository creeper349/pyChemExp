import numpy as np
import math
from interpolation import *

def first_derivative(x:np.ndarray,
                     y:np.ndarray,
                     interpolated=True,
                     method='central')->np.ndarray:
    """calculate first-order derivative of given discrete function y=f(x)

    Args:
        x (np.ndarray): values of independent variable
        y (np.ndarray): values of dependent variable
        interpolated (bool, optional): if true, cubic interpolation will be processed before calculating derivatization. Defaults to True.
        method (str, optional): 'backward', 'central' or 'forward'. Defaults to 'central'.

    Returns:
        np.ndarray: 1st derivative of given function
    """
    
    if interpolated:
        x_,y_,_=interpolation(x,y)
    else:
        x_,y_=x,y
    dev=np.zeros((len(x_)))
        
    dev[0]=(y_[1]-y_[0])/(x_[1]-x_[0])
    dev[-1]=(y_[-1]-y_[-2])/(x_[-1]-x_[-2])
    i=1
    while i<len(x_)-1:
        if method=='backward':
            dev[i]=(y_[i]-y_[i-1])/(x_[i]-x_[i-1])
        elif method=='forward':
            dev[i]=(y_[i+1]-y_[i])/(x_[i+1]-x_[i])
        elif method=='central':
            dev[i]=(y_[i+1]-y_[i-1])/(x_[i+1]-x_[i-1])
        else:
            pass
        i+=1
    return dev

def multi_derivative(x:np.ndarray,
                     y:np.ndarray,
                     degree:int,
                     interpolated=True)->np.ndarray:
    """calculate n-th derivative of given discrete function y=f(x)

    Args:
        x (np.ndarray): values of independent variable
        y (np.ndarray): values of dependent variable
        degree (int): degree of derivate, n for f^{(n)}(x)
        interpolated (bool, optional): if true, cubic interpolation will be processed before calculating derivatization.. Defaults to True.

    Returns:
        np.ndarray: n-th derivative of given function
    """
    if interpolated:
        x_,y_,_=interpolation(x,y)
    else:
        x_,y_=x,y
    dev=np.zeros((degree+1,len(x_)))
    dev[0]=y_
    h=np.zeros((len(x_)+1,))
    h[1:-1]=[x_[i]-x_[i-1] for i in range(1,len(x_))]
    h[0],h[-1]=h[1],h[-2]
    x_,y_=np.r_[2*x_[0]-x_[1],x_,2*x_[-1]-x_[-2]],np.r_[2*y_[0]-y_[1],y_,2*y_[-1]-y_[-2]]
    
    for i in range(1,degree+1):
        j,sum=0,np.zeros(len(h)-1)
        for j in range(len(h)-1):
            for k in range(i):
                sum[j]+=((h[j+1]**k+(-1)**(i+k)*h[j]**k)/math.factorial(k))*dev[k,j]
        for j in range(len(h)-1):
            dev[i,j]=(y_[j+2]+(-1)**i*y_[j]-sum[j])*math.factorial(i)/(h[j]**i+h[j+1]**i)
            # derive n-th derivative from taylor series of given function
            
    return dev[-1]

# test code:

import matplotlib.pyplot as plt
a=np.array([1,2,3,4,5,6,7,8,9,11,16])
b=np.array([1,3,6,10,15,20,17,14,9,4,8])
plt.scatter(a,b,color='red')
x,y,_=interpolation(a,b)
plt.plot(x,y,color='red')
plt.plot(x,multi_derivative(x,y,2,False),color='green')
plt.show()