import numpy as np
import warnings
from enum import Enum,auto
from .devint import intergration_func,INTEGRATION_TYPE

class INTERPOLATE_TYPE(Enum):
    LINEAR=auto()
    CUBIC=auto()
    LAGRANGE=auto()
    
class Interpolation:
    def __init__(self,
                 x:np.ndarray,
                 y:np.ndarray,
                 interpo_type:INTERPOLATE_TYPE,
                 *args,**kwargs):
        """create a interpolated curve object

        Args:
            x (np.ndarray): indenpendent variable (must arranged from small number to large number)
            y (np.ndarray): dependent variable to x
            interpo_type (INTERPOLATE_TYPE): expected interpolation type
        """
        if len(x)!=len(y):
            raise ValueError("x and y are not in the same length.")
        if not np.all(np.diff(x))>0:
            raise ValueError("x should be monotonically increasing.")
        self.x=x
        self.y=y
        self.length=len(self.x)
        self.def_domain=(x[0],x[-1])
        self.interpolation_type=interpo_type
        self.para=None
        if self.interpolation_type==INTERPOLATE_TYPE.CUBIC:
            self._cubicinterpo()
        elif self.interpolation_type==INTERPOLATE_TYPE.LAGRANGE and self.length>50:
            warnings.warn("If there are too many points, Lagarange interpolation is not recommended.")
            
    def __call__(self, val:float, *args, **kwargs):
        if val<self.x[0] or val>self.x[-1]:
            raise ValueError("Value out of range.")
        elif val==self.x[-1]:
            return self.y[-1]
        else:
            index=np.searchsorted(self.x,val,"right")-1
            match self.interpolation_type:
                case INTERPOLATE_TYPE.LINEAR:
                    return self.y[index]+(self.y[index+1]-self.y[index])/(self.x[index+1]
                            -self.x[index])*(val-self.x[index])
                case INTERPOLATE_TYPE.CUBIC:
                    return self.para[4*index]*val**3+self.para[4*index+1]*val**2+\
                        self.para[4*index+2]*val+self.para[4*index+3]
                case INTERPOLATE_TYPE.LAGRANGE:
                    func=0
                    for j in range(self.length):
                        L_j=1
                        for i in range(self.length):
                            if i!=j:
                                L_j=L_j*(val-self.x[i])/(self.x[j]-self.x[i])
                        func+=L_j*self.y[j]
                    return func
                
    def __len__(self):
        return self.length
    
    def __getitem__(self,index:int):
        if index>=self.length or index<0:
            raise ValueError("Index out of range.")
        else:
            return (self.x[index],self.y[index])
    
    def plotting(self,ax,
                 num_points:int=None,
                 derivative:bool=False,
                 epsilon:float=1e-5,
                 *args,**kwargs):
        if not num_points:
            num_points=10*self.length
        if derivative:
            x_=np.linspace(self.x[0],self.x[-1]-epsilon*(self.x[-1]-self.x[-2]),num_points)
            y_=np.array([self.derivative(x_[i]) for i in range(num_points)])
        else:
            x_=np.linspace(self.x[0],self.x[-1],num_points)
            y_=np.array([self(x_[i]) for i in range(num_points)])
        line, = ax.plot(x_,y_,*args,**kwargs)
        ax.figure.canvas.draw()
        ax.legend()
        return line
    
    def scattering(self,ax,*args,**kwargs):
        points=ax.scatter(self.x,self.y,*args,**kwargs)
        ax.figure.canvas.draw()
        ax.legend()
        return points
    
    def derivative(self,val:float,*args,**kwargs):
        if val<self.x[0] or val>=self.x[-1]:
            raise ValueError("Value out of range.")
        else:
            index=np.searchsorted(self.x,val,"right")-1
            match self.interpolation_type:
                case INTERPOLATE_TYPE.LINEAR:
                    return (self.y[index+1]-self.y[index])/(self.x[index+1]-self.x[index])
                case INTERPOLATE_TYPE.CUBIC:
                    return 3*self.para[4*index]*val**2+2*self.para[4*index+1]*val+\
                        self.para[4*index+2]
                case INTERPOLATE_TYPE.LAGRANGE:
                    L=0
                    for j in range(self.length):
                        L_j=0
                        for k in range(self.length):
                            if k!=j:
                                mul=1
                                for m in range(self.length):
                                    if m!=j and m!=k:
                                        mul=mul*(val-self.x[m])/(self.x[j]-self.x[m])
                                L_j+=mul/(self.x[j]-self.x[k])
                        L+=self.y[j]*L_j
                    return L
                
    def integration(self,lb:float,ub:float,*args,**kwargs):
        if not (lb>=self.x[0] and lb<self.x[-1] and ub>self.x[0] \
            and ub<=self.x[-1] and lb<ub):
            raise ValueError("Value out of range.")
        else:
            lb_domain=np.searchsorted(self.x,lb,side="right")-1
            ub_domain=np.searchsorted(self.x,ub,side="right")-1
            match self.interpolation_type:
                case INTERPOLATE_TYPE.LINEAR:
                    id_l=lb_domain+1
                    integ=0.5*(self.y[id_l]+self(lb))*(self.x[id_l]-lb)
                    while id_l<ub_domain:
                        integ+=0.5*(self.y[id_l+1]+self.y[id_l])*(self.x[id_l+1]-self.x[id_l])
                        id_l+=1
                    integ+=0.5*(self(ub)+self.y[id_l])*(ub-self.x[id_l])
                    return integ
                case INTERPOLATE_TYPE.CUBIC:
                    id_l=lb_domain+1
                    integ=0.25*self.para[4*id_l-4]*(self.x[id_l]**4-lb**4)+\
                        1/3*self.para[4*id_l-3]*(self.x[id_l]**3-lb**3)+\
                            0.5*self.para[4*id_l-2]*(self.x[id_l]**2-lb**2)+\
                                self.para[4*id_l-1]*(self.x[id_l]-lb)
                    while id_l<ub_domain:
                        integ+=0.25*self.para[4*id_l]*(self.x[id_l+1]**4-self.x[id_l]**4)+\
                        1/3*self.para[4*id_l+1]*(self.x[id_l+1]**3-self.x[id_l]**3)+\
                            0.5*self.para[4*id_l+2]*(self.x[id_l+1]**2-self.x[id_l]**2)+\
                                self.para[4*id_l+3]*(self.x[id_l+1]-self.x[id_l])
                        id_l+=1
                    integ+=0.25*self.para[4*id_l]*(ub**4-self.x[id_l]**4)+\
                        1/3*self.para[4*id_l+1]*(ub**3-self.x[id_l]**3)+\
                            0.5*self.para[4*id_l+2]*(ub**2-self.x[id_l]**2)+\
                                self.para[4*id_l+3]*(ub-self.x[id_l])
                    return integ
                case INTERPOLATE_TYPE.LAGRANGE:
                    return intergration_func(lb,ub,self,
                                             intergral_type=INTEGRATION_TYPE.BOOLE,
                                             n_domains=ub_domain-lb_domain,
                                             absolute=False)
                                    
    def _cubicinterpo(self,
                  boundries:tuple=(0,0)):
        
        x,y=self.x, self.y
        legh,i=len(x),0
        mat=np.zeros((4*legh-4,4*legh-4))
        sol=np.zeros((4*legh-4,))
        while i<legh-1:
            mat[i,4*i]=x[i]**3
            mat[i,4*i+1]=x[i]**2
            mat[i,4*i+2]=x[i]
            mat[i,4*i+3]=1
            sol[i]=y[i]
        
            mat[legh-1+i,4*i]=x[i+1]**3
            mat[legh-1+i,4*i+1]=x[i+1]**2
            mat[legh-1+i,4*i+2]=x[i+1]
            mat[legh-1+i,4*i+3]=1
            sol[legh-1+i]=y[i+1]
        
            if i==0:
                mat[3*legh-4+i,4*i],mat[3*legh-4+i,i+1]=6*x[i],2
                sol[3*legh-4+i]=boundries[0]
            else:
                if i==legh-2:
                    mat[3*legh-3+i,4*i],mat[3*legh-3+i,i+1]=6*x[i+1],2
                    sol[3*legh-3+i]=boundries[1]
                 
                mat[3*legh-4+i,4*i]=6*x[i]
                mat[3*legh-4+i,4*i+1]=2
                mat[3*legh-4+i,4*i-4]=-6*x[i]
                mat[3*legh-4+i,4*i-3]=-2
                sol[3*legh-4+i]=0
              
                mat[2*legh-3+i,4*i]=3*x[i]**2
                mat[2*legh-3+i,4*i+1]=2*x[i]
                mat[2*legh-3+i,4*i+2]=1
                mat[2*legh-3+i,4*i-4]=-3*x[i]**2
                mat[2*legh-3+i,4*i-3]=-2*x[i]
                mat[2*legh-3+i,4*i-2]=-1
                sol[2*legh-3+i]=0
            
            i+=1
        
        self.para=np.linalg.solve(mat,sol)
        # cubic curve stabilization finishes
        # parameters are solved to be this form:
        # [a_1 b_1 c_1 d_1 a_2 b_2 c_2 d_2 ... a_n b_n c_n d_n]
        # where a_i to d_i is cubic polynomial coefficients on each part of curve
        # y=a_i*x^3+b_i*x^2+c_i*x+d_i (x_i<x<x_{i+1},i=0,1,...,n), n+1 is total number of points