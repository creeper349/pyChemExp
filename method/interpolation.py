import numpy as np

def interpolation(x:np.ndarray,
                  y:np.ndarray,
                  sample_dots:int=10,
                  boundries:tuple=(0,0))->np.ndarray:
    """cubic curve interpolation

    Args:
        x (np.ndarray): 1D array, independent variable
        y (np.ndarray): 1D array, dependent variable
        sample_dots (int): how many points are sampled between two original data points
        boundries (tuple): assign values of f''(x_0) and f''(x_n) to make solution of the function unique

    Returns:
        np.ndarray: interpolated curve, more points are inserted between original data points to
        make the curve smooth
    """
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
        
    para=np.linalg.inv(mat).dot(sol)
    # cubic curve stabilization finishes
    
    int_x_list,int_y_list,i=[],[],0
    while i<legh-1:
        step=(x[i+1]-x[i])/(sample_dots+1)
        j=x[i]
        while j<x[i+1]:
            int_x_list.append(j)
            int_y_list.append(para[4*i]*j**3+para[4*i+1]*j**2+\
                para[4*i+2]*j+para[4*i+3])
            j+=step
        i+=1
    
    int_x_list.append(x[-1])
    int_y_list.append(para[-4]*x[-1]**3+para[-3]*x[-1]**2+para[-2]*x[-1]+para[-1])    
    int_x_list=np.array(int_x_list)
    int_y_list=np.array(int_y_list)
    
    return int_x_list,int_y_list

# test code:
'''
import matplotlib.pyplot as plt
a=np.array([1,2,3,4,5,6,7,8,9,11,16])
b=np.array([1,3,6,10,15,20,17,14,9,4,8])
plt.scatter(a,b,color='red')
x,y=interpolation(a,b)
plt.plot(x,y)
plt.show()'''