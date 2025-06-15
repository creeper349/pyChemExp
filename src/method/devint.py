import numpy as np
from enum import Enum, auto

def derivative_discrete(x:np.ndarray,
                        y:np.ndarray,
                        *args,**kwargs):
    if len(x)!=len(y):
            raise ValueError("x and y are not in the same length.")
    if not np.all(np.diff(x))>0:
            raise ValueError("x should be monotonically increasing.")
    dev=np.zeros((len(x),),dtype=float)
    try:
        for _ in range(len(x)):
            if _==0:
                dev[_]=(y[_+1]-y[_])/(x[_+1]-x[_])
            elif _==len(x)-1:
                dev[_]=(y[_]-y[_-1])/(x[_]-x[_-1])
            else:
                h1=x[_]-x[_-1]
                h2=x[_+1]-x[_]
                A, B, C=-h2/(h1*(h1 + h2)), h1/(h2*(h1 + h2)), (-h1 + h2)/(h1*h2)
                dev[_]=A*y[_-1]+B*y[_+1]+C*y[_]
        return dev
    except:
        raise RuntimeError("Error occurs when computing derivative of array x.")
    
def integration_discrete(x:np.ndarray,
                         y:np.ndarray,
                         absolute=False,
                         *args,**kwargs):
    if len(x)!=len(y):
        raise ValueError("x and y are not in the same length.")
    if not np.all(np.diff(x))>0:
        raise ValueError("x should be monotonically increasing.")
    area=0.0
    try:
        for i in range(len(x)-1):
            if absolute:
                area+=0.5*(abs(y[i])+abs(y[i+1]))*(x[i+1]-x[i])
            else:
                area+=0.5*(y[i+1]+y[i])*(x[i+1]-x[i])
        return area
    except:
        raise RuntimeError("Error occurs when computing integration of array x.")

class DERIVATIZATION_TYPE(Enum):
    FORWARD=auto() #o(h)
    BACKWARD=auto() #o(h)
    CENTRAL=auto() #o(h^2)
    FOUR_POINTS=auto() #o(h^4)
    SIX_POINTS=auto() #o(h^6)
    
def derivative_func(val:float,
                    func:callable,
                    dev_type:DERIVATIZATION_TYPE=DERIVATIZATION_TYPE.CENTRAL,
                    h=None,
                    epsilon=None,
                    *args, **kwargs):
    if not callable(func):
        raise TypeError("Parameter func should be a callable function.")
    if not epsilon:
        epsilon=np.finfo(float).eps
    try:
        match dev_type:
            case DERIVATIZATION_TYPE.FORWARD:
                # automatically adjust h to h=\epsilon^\frac{1}{p+1}*max(|x|,1)
                # where p is order of accuracy of derivative o(h^p)
                if not h:
                    h=epsilon**0.5*max(abs(val),1)
                return (func(val+h)-func(val))/h
            case DERIVATIZATION_TYPE.BACKWARD:
                if not h:
                    h=epsilon**0.5*max(abs(val),1)
                return (func(val)-func(val-h))/h
            case DERIVATIZATION_TYPE.CENTRAL:
                if not h:
                    h=epsilon**(1/3)*max(abs(val),1)
                return (func(val+h)-func(val-h))/(2*h)
            case DERIVATIZATION_TYPE.FOUR_POINTS:
                if not h:
                    h=epsilon**0.2*max(abs(val),1)
                return (-func(val+2*h)+8*func(val+h)-8*func(val-h)+func(val-2*h))/(12*h)
            case DERIVATIZATION_TYPE.SIX_POINTS:
                if not h:
                    h=epsilon**(1/7)*max(abs(val),1)
                return (func(val-3*h)-9*func(val-2*h)+45*func(val-h)-45*func(val+h)+\
                    9*func(val+2*h)-func(val+3*h))/(60*h)
    except:
        raise RuntimeError(f"Invalid value {val} for defined function.")

class INTEGRATION_TYPE(Enum):
    RECTANGLE=auto()
    TRAPZOID=auto()
    BOOLE=auto()

def intergration_func(lb:float,ub:float,
                      func:callable,
                      intergral_type:INTEGRATION_TYPE=INTEGRATION_TYPE.TRAPZOID,
                      n_domains:int=100,
                      absolute=False,
                      *args,**kwargs):
    if ub<=lb:
        raise ValueError("Upper bound must be larger than lower bound.")
    if not callable(func):
        raise TypeError("Parameter func should be a callable function.")
    step=(ub-lb)/n_domains
    x,area,i=lb,0,0
    while i<n_domains:
        x=lb+i*step
        try:
            match intergral_type:
                case INTEGRATION_TYPE.RECTANGLE:
                    area+=step*abs(func(x+step/2)) if absolute\
                        else step*func(x+step/2)
                case INTEGRATION_TYPE.TRAPZOID:
                    area+=0.5*step*(abs(func(x))+abs(func(x+step))) if absolute\
                        else 0.5*step*(func(x)+func(x+step))
                case INTEGRATION_TYPE.BOOLE:
                    area+=step/90*(7*abs(func(x))+32*abs(func(x+0.25*step))+\
                        12*abs(func(x+0.5*step))+32*abs(func(x+0.75*step))+7*abs(func(x+step)))\
                        if absolute else step/90*(7*func(x)+32*func(x+0.25*step)+\
                        12*func(x+0.5*step)+32*func(x+0.75*step)+7*func(x+step))
                    # Boole's Rule for numerical integration calculation
                    # int_a^bf(x)\text{d}x=\frac{2h}{45}(7*f(x_0)+32*f(x_1)+12*f(x_2)+7*f(x_3))
                    # x_0, x_1, x_2, x_3 \in [a,b] and are uniform (x_1-x_0=x_2-x_1=...)
                    # 5-th order accuracy
        except:
            raise RuntimeError(f"Invalid value {x} for the defined function.")
        i+=1
    return area