import numpy as np

def double_log(x:float,y:float):
    try:
        return np.log(x),np.log(y)
    except:
        raise ValueError("Invalid input of transformation \"log\"")

def double_reciprocal(x:float,y:float):
    try:
        return 1/x,1/y
    except:
        raise ValueError("Invalid input of reciprocal transformation")
    
def firstorder_kinetics(t:float,c:float):
    try:
        return t,np.log(c)
    except:
        raise ValueError("Invalid input")
    
def secondorder_kinetics(t:float,c:float):
    try:
        return t,1/c
    except:
        raise ValueError("Invalid input")
    
def hanes_woolf(S:float,V:float):
    try:
        return S,S/V
    except:
        raise ValueError("Invalid input")
    
def Arrhenius(T:float,k:float):
    try:
        return 1/T,np.log(k)
    except:
        raise ValueError("Invalid input")
    
def hyperbolic(x,a,b,c):
    return a*x**2+b*x+c

def expdec(x,a,b,c):
    return a+np.exp(-b*x+c)

def hyperbl(x,a,b):
    try:
        return a*x/(x+b)
    except:
        raise ValueError("Invalid input")
    
def rational1(x,a,b,c):
    try:
        return (a*x+b)/(1+c*x)
    except:
        raise ValueError("Invalid input")
    
def rational2(x,a,b,c,d):
    try:
        return (a*x+b)/(1+c*x+d*x**2)
    except:
        raise ValueError("Invalid input")
    
def rlogistic(x,a,b,c):
    return a/(1+b*np.exp(-c*x))

def cubic(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

def holliday(x,a,b,c):
    try:
        return a/(1+b*x+c*x**2)
    except:
        raise ValueError("Invalid input")
    
def logrithm(x,a,b):
    try:
        return a+np.log(x+b)
    except:
        raise ValueError("Invalid input")