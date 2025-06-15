import numpy as np
import inspect

class Regressor:
    def __init__(self,func:callable,
                   x:np.ndarray,
                   y:np.ndarray,
                   initial_para:np.ndarray=None,
                   max_iter:int=100,
                   tol:float=1e-5,
                   bound:tuple=(-1,1),
                   theta:np.ndarray=None,
                   grad_func:callable=None,
                   *args,**kwargs):
        """linear or non-linear regression depending on a given function model on data x and y

        Args:
            func (callable): function model
            x (np.ndarray): 1D array, independent variable
            y (np.ndarray): 1D array, dependent variable
            initial_para (np.ndarray, optional): initial guess for parameters in the function model. Defaults to None.
            max_iter (int): parameters will be updated up to `max_iter` times.
            bound (tuple): set a domain to choose initialized parameters.
            theta (float, optional): small change made to parameters when obtaining a gradient. Defaults to None (auto-adaptive).
            grad_func (optional): function to compute gradient of L2 loss when using the given model. \
                If it is None, gradient will be obtained by numerical central derivative.
            **kwargs: super-parameters used in gradient descent methods can be listed here
        """
        if len(x)!=len(y):
            raise ValueError("x and y are not in the same length.")
        self.x,self.y=x,y
        self.func=func
        sig=inspect.signature(func)
        self.parameters = list(sig.parameters.values())[1:]
        if initial_para is None:
            self.parameters = bound[0]+(bound[1]-bound[0])*np.random.rand(len(self.parameters))
        else:
            if len(initial_para)!=len(self.parameters):
                raise ValueError("Invalid initial parameters.")
            else:
                self.parameters = initial_para
        self.max_iter=max_iter
        self.tol=tol
        self.theta=theta
        self.grad_func=grad_func
        
    def fit(self,lambda_modifier:float=2.0,
            lambda_init:float=0.1,
            *args,**kwargs):
        try:
            lamb=lambda_init
            loss_old=np.inf
            jac=None
            for _ in range(self.max_iter):
                if self.grad_func is None:
                    dtheta, residues, jac=_grad_func(self.func,self.x,self.y,self.parameters,lamb)
                else:
                    dtheta=self.grad_func(self.func,self.x,self.y,self.parameters,*args,**kwargs)
                    residues=self.func(self.x)-self.y
                loss=0.5*np.sum(residues**2)
                self.parameters += dtheta
                if loss>=loss_old:
                    lamb = min(lamb * lambda_modifier, 1e6)
                else:
                    lamb = max(lamb / lambda_modifier, 1e-12)
                loss_old=loss
                if lamb<self.tol:
                    break
            if jac is None:
                jac=_compute_jac(self.func,self.x,self.y,self.parameters,1e-8)
            sigma2=np.sum((self(self.x)-self.y)**2)/(len(self.x)-len(self.parameters))
            pcov=sigma2*np.linalg.pinv(jac.T.dot(jac))
            return self.parameters, pcov
        except:
            raise RuntimeError("Failed to fit a regression curve. Sometimes it is because of some\
                problems in random parameter initilization. You could try again.")
    
    def __call__(self, x, *args, **kwds):
        try:
            return self.func(x,*self.parameters)
        except:
            raise ValueError("Invalid input.")
        
    def plot(self,ax,num_points:int=None,*args,**kwargs):
        if not num_points:
            num_points=len(self.x)*10
        x=np.linspace(min(self.x),max(self.x),num_points)
        y=self(x)
        line,=ax.plot(x,y,*args,**kwargs)
        ax.figure.canvas.draw()
        ax.legend()
        return line
    
    def scatter(self,ax,*args,**kwargs):
        dots=ax.scatter(self.x,self.y,*args,**kwargs)
        return dots
        
def _grad_func(func:callable,
               x:np.ndarray,
               y:np.ndarray,
               parameters,
               lamb:float,
               theta:np.ndarray=None,
               *args,**kwargs):
    if theta is None:
        epsilon=np.finfo(float).eps
        theta=epsilon**(1/3)*(np.abs(parameters) + 1e-8)
    residues=np.zeros(len(x),dtype=float)
    jac=_compute_jac(func,x,y,parameters,theta)
    residues=func(x,*parameters)-y
    return np.linalg.pinv(jac.T.dot(jac)+lamb*np.diag(jac.T.dot(jac))).\
                           dot(jac.T).dot(-residues), residues, jac
                           
def _compute_jac(func,x,y,parameters,theta):
    jac=np.zeros((len(x),len(parameters)),dtype=float)
    for i in range(len(x)):
        for j in range(len(parameters)):
            h1,h2=parameters.copy(),parameters.copy()
            h1[j]+=theta[j]
            h2[j]-=theta[j]
            jac[i,j]=(func(x[i],*h1)-func(x[i],*h2))/(2*theta[j])
    return jac