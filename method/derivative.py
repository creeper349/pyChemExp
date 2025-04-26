import numpy as np
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

