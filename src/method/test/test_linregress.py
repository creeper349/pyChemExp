import numpy as np
import matplotlib.pyplot as plt
from ..regression import SAPARATE_DELETE, LinRegressor

if __name__ == '__main__':
    
    noise=np.random.randn(20)
    x=np.linspace(-10,10,20)
    y=3.5*x+4.2+noise
    x=np.append(x,np.array([2.2,-3.4,-9.8]))
    y=np.append(y,np.array([28.2,19.6,3.3]))
    regressor=LinRegressor(x,y,del_saparated_point=True,optim_method=SAPARATE_DELETE.Zscore,
                       threshold=1.5,max_iter=100)

    print(regressor.fit(num_samples=3))
    fig,ax=plt.subplots()
    regressor.plot(ax)
    regressor.scatter(ax)
    plt.show()
    
    x1=np.linspace(-10,10,20)
    y1=[3.5*x1+4.2+np.random.randn(20),3.5*x1+4.2+np.random.randn(20),3.5*x1+4.2+np.random.randn(20)]
    y1=np.array(y1)
    regressor=LinRegressor(x1,y1,del_saparated_point=False)
    print(regressor.fit(use_weights=True))
    print(regressor.weights)
    fig,ax=plt.subplots()
    regressor.plot(ax)
    regressor.errorbar(ax)
    plt.show()