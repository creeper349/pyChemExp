import numpy as np

class PolynomialRegressor:
    def __init__(self,
                 x:np.ndarray,
                 y:np.ndarray,
                 degree:int=2,
                 lamb:float=0.0,
                 *args,**kwargs):
        """Polynominal regression to x and y

        Args:
            x (np.ndarray): 1D array, independent variable
            y (np.ndarray): 1D array, dependent variable
            degree (int, optional): Degree of the polyniomial model. Defaults to 2.
            lamb (float, optional): L2 regularization term coefficient. Defaults to 0.0.

        """
        if len(x)!=len(y):
            raise ValueError("x and y are not in the same length.")
        self.x,self.y=x,y
        self.lamb=lamb
        self.degree=degree
        
    def fit(self):
        """fit the polynominal model

        Returns:
            weights: Weights on each x^k term
            pcov: Covarience matrix of parameters
            Rsq: correlation coefficient
            Rsq_adj: adjusted correlation coefficient
        """
        poly=np.zeros((len(self.x),self.degree+1),dtype=float)
        for _ in range(self.degree+1):
            poly[:,_]=self.x**_
        self.weights=np.linalg.solve(poly.T.dot(poly)+self.lamb*np.eye(self.degree+1),
                                     poly.T.dot(self.y))
        sigma2=np.sum((self(self.x)-self.y)**2)/(len(self.x)-self.degree-1)
        self.pcov=sigma2*np.linalg.inv(poly.T.dot(poly)+self.lamb*np.eye(self.degree+1)).\
            dot(poly.T.dot(poly)).dot(np.linalg.inv(poly.T.dot(poly)+self.lamb*np.eye(self.degree+1)))
        self.Rsq,self.Rsq_adj=_compute_Rsq(self.x,self.y,self(self.x),self.degree+1)
        return self.weights,self.pcov,self.Rsq,self.Rsq_adj
        
    def __call__(self, x, *args, **kwds):
        try:
            x_arr = np.asarray(x)
            if x_arr.ndim == 0:
                x_arr = x_arr.reshape(1)
            X = np.vander(x_arr, N=self.degree + 1, increasing=True)
            return X.dot(self.weights)
        except:
            raise RuntimeError("Invalid input or undefined parameters (use LinRegressor.fit() first)")
        
    def plot(self,ax,decimals=4,num_points=None,*args,**kwargs):
        """draw the regression curve.

        Args:
            ax : matplotlib canvas object
            decimals (int, optional): For printing the expression of the curve,\
                how many decimals will be retained. Defaults to 4.

        Returns:
            line: regression curve, a matplotlib Line2D object
        """
        expr="$y="
        for i in range(self.degree+1):
            if i==0:
                expr+=rf"{np.round(self.weights[i],decimals)}"
            elif i==1:
                sign="+" if self.weights[i]>0 else "-"
                expr+=rf"{sign}{np.round(abs(self.weights[i]),decimals)}x"
            else:
                sign="+" if self.weights[i]>0 else "-"
                expr+=rf"{sign}{np.round(abs(self.weights[i]),decimals)}x^{i}"
        expr+=rf"$"
        legend=kwargs.pop("legend",expr)
        if num_points is None:
            num_points=len(self.x)*10
        x=np.linspace(min(self.x),max(self.x),num_points)
        y=self(x)
        line, =ax.plot(x,y,label=legend,*args,**kwargs)
        ax.figure.canvas.draw()
        ax.legend()
        return line
    
    def scatter(self,ax,*args,**kwargs):
        dots=ax.scatter(self.x,self.y,*args,**kwargs)
        return dots
    
class PolyvarRegressor:
    def __init__(self,
                 x:np.ndarray,
                 y:np.ndarray,
                 lamb:float=0.0):
        """fit a linear model for multiple variables

        Args:
            x (np.ndarray): 2D array (n_samples*n_dim)
            y (np.ndarray): 1D array
            lamb (float, optional): L2 regularization term coefficient. Defaults to 0.0.
        """
        if x.shape[0]!=len(y):
            raise ValueError("x and y are not in the same length.")
        self.x,self.y,self.lamb=x,y,lamb
        
    def fit(self):
        x=np.concatenate((np.ones((self.x.shape[0],1)),self.x),axis=1)
        I=np.eye(x.shape[1])
        I[0,0]=0
        self.weights=np.linalg.solve(x.T.dot(x)+self.lamb*I,x.T.dot(self.y))
        sigma2=np.sum((self.y-self(x))**2)/(self.x.shape[0]-self.x.shape[1])
        self.pcov=sigma2*np.linalg.inv(x.T.dot(x)+self.lamb*I).\
            dot(x.T.dot(x)).dot(np.linalg.inv(x.T.dot(x)+self.lamb*I))
        self.Rsq,self.Rsq_adj=_compute_Rsq(x,self.y,self(x),num_para=x.shape[1])
        return self.weights,self.pcov,self.Rsq,self.Rsq_adj
        
    def __call__(self,x,*args,**kwargs):
        if len(x.shape)==1:
            x.reshape(1,len(x))
        if not x.shape[1]==self.x.shape[1]+1:
            x=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        return x.dot(self.weights)
        
def _compute_Rsq(x,y,y_pred,num_para):
    SSres=np.sum((y-y_pred)**2)
    SStot=np.sum((y-np.mean(y))**2)
    Rsq=1-SSres/SStot
    Rsq_adj=1-((1-Rsq)*(len(x)-1)/(len(x)-num_para-1))
    return Rsq,Rsq_adj