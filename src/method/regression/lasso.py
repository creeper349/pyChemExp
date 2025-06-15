import numpy as np

def LASSO(X:np.ndarray,y:np.ndarray,
          C:float=1.0,
          learning_rate:float=0.001,
          tol=1e-7,
          max_iter=1000):
    """Lasso regression function

    Args:
        X (np.ndarray): 2D array
        y (np.ndarray): 1D array
        C (float, optional): L1 regularization term coefficient. Defaults to 1.0.
        learning_rate (float, optional): Learning rate in gradient descent. Defaults to 0.001.
        tol (_type_, optional): When \Delta w < tol, parameter updating will be stopped. Defaults to 1e-7.
        max_iter (int, optional): When parameter updating occurs more than max_iter times,\
            the function will be stopped. Defaults to 1000.

    Returns:
        w, b: sparse regression weights and bias
    """
    
    w = np.zeros((X.shape[1])).reshape(X.shape[1],1)
    b = 0.0
    for i in range(max_iter):
        grad_w = 2 * np.dot(X.T,np.dot(X,w)-y) / X.shape[0]
        grad_b = 2 * np.sum(np.dot(X,w)+b-y) / X.shape[0]
        
        w_temp=w-learning_rate*grad_w

        threshold = C * learning_rate
        w_new = np.sign(w_temp) * np.maximum(np.abs(w_temp) - threshold, 0)
        b_new = b - learning_rate * grad_b
        
        if np.linalg.norm(w_new - w) < tol and abs(b_new - b) < tol:
            break
        w = w_new
        b = b_new
    return w, b