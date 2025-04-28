import numpy as np
import inspect

def regression(func:callable,
               x:np.ndarray,
               y:np.ndarray,
               initial_para:np.ndarray=None,
               method:str='normal',
               theta:float=1e-6,
               **kwargs):
    """linear or non-linear regression depending on a given function model on data x and y

    Args:
        func (callable): function model
        x (np.ndarray): 1D array, independent variable
        y (np.ndarray): 1D array, dependent variable
        initial_para (np.ndarray, optional): initial guess for parameters in the function model. Defaults to None.
        method (str, optional): 'normal', 'momentum', 'RMSProp' or 'Adam'. Choice for gradient descent methods. Defaults to 'normal'.
        theta (float, optional): small change made to parameters when obtaining a gradient. Defaults to 1e-6.
        **kwargs: super-parameters used in gradient descent methods can be listed here
    """
    sig=inspect.signature(func)
    parameters = list(sig.parameters.values())
    if not(np.array_equal(initial_para,None)):
        if len(initial_para)==len(parameters[1:]):
            parameters[1:]=initial_para.astype(float)
        else:
            parameters[1:]=np.random.rand(len(parameters[1:]))
    else:
        parameters[1:]=np.random.rand(len(parameters[1:]))
    # extract parameters and initialize their values
    
    grad=np.zeros(len(parameters[1:]))
    thetas=np.eye(len(parameters[1:]))*theta
    for i in range(len(x)):
        for j in range(1,len(parameters)):
            args1=[x[i]]+np.add(parameters[1:],thetas[j-1]).tolist()
            args2=[x[i]]+np.add(parameters[1:],-thetas[j-1]).tolist()
            grad[j-1]+=(func(*args1)-func(*args2))/(2*theta)
    print(grad)
    # obtaining gradient
    
    
def linear(x,a,b):
    return a*x+b

def exponent(x,a,b,c):
    return a*np.exp(b*x)+c

def logarithm(x,a,b,c):
    return a*np.log(x+b)+c

regression(linear,[0.003088326,0.00304683,0.003002101,\
    0.002957792,0.002920731,0.002879355],[10.28021019,10.50397105,10.73900197,\
        10.94799558,11.14012114,11.34391492])