import numpy as np

def _cubic_regression(x:np.ndarray,y:np.ndarray,sample_dots:int,selected=[1,2]):
    # used for curve smoothify
    # input two points and its neighbour points at two sides
    # create a cubic function according to input value x and y (4 elements)
    # return points (number is sample_dots) on calibrated curve between selected two points
    mat=np.zeros((4,4))
    mat[:,0]=x**3
    mat[:,1]=x**2
    mat[:,2]=x
    mat[:,3]=np.ones((4,))
    para=np.linalg.inv(mat).dot(y)
    sample_dots_x,sample_dots_y=np.linspace(x[selected[0]],x[selected[1]],sample_dots),[]
    for dot in sample_dots_x:
        sample_dots_y.append(para[0]*dot**3+para[1]*dot**2+para[2]*dot+para[3])
    return np.array(sample_dots_y)

def _cubic_smooth(x:np.ndarray,y:np.ndarray,sample_dots:int=None):
    # aquire derivative of y(x) after linking the points by local cubic curve
    if sample_dots==None:
        sample_dots=10*(len(x)-1)
    _sample_dots=int(sample_dots/(len(x)-1))
    i,end,sample_dots_x_list,sample_dots_y_list=0,len(x)-1,[],[]
    while i<end:
        if i==0:
            s_dots=_cubic_regression(x[0:4],y[0:4],_sample_dots,selected=[0,1])
        elif i==end-1:
            s_dots=_cubic_regression(x[-4:],y[-4:],_sample_dots,selected=[2,3])
        else:
            s_dots=_cubic_regression(x[i-1:i+3],y[i-1:i+3],_sample_dots)
        sample_dots_x_list.append(np.linspace(x[i],x[i+1],_sample_dots))
        sample_dots_y_list.append(s_dots)
        i+=1
        
    sample_dots_x_list,sample_dots_y_list=np.array(sample_dots_x_list),np.array(sample_dots_y_list)
    sample_dots_x_list=sample_dots_x_list.reshape((sample_dots,))
    sample_dots_y_list=sample_dots_y_list.reshape((sample_dots,))
    return sample_dots_x_list,sample_dots_y_list
    
import matplotlib.pyplot as plt
a=np.array([1,2,3,4,5,6,7,8,9,11,16])
b=np.array([1,3,6,10,15,20,17,14,9,4,8])
plt.scatter(a,b,color='red')
x,y=_cubic_smooth(a,b)
plt.plot(x,y)
plt.show()