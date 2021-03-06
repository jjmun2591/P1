import numpy as np
np.random.seed(1)

class lin_reg():
    
    def __init__(self,trainDataX,trainDataY,hasIntercept=False):
        self.hasIntercept = hasIntercept
        if hasIntercept:
            self.theta = np.random.rand(1,input.shape[1]+1)
        else:
            self.theta = np.random.rand(1,input.shape[1])
        self.trainDataX = trainDataX
        self.trainDataY = trainDataY


    def predict(self,X):
        if self.hasIntercept:            
            y = np.dot(X,self.theta[:,1:].T) + self.theta[:,0]
        else:
            y = np.dot(X,self.theta.T)
        return y
    

    def gradientDesc(self,alpha,epochs):
        theta = self.theta
        x = self.trainDataX
        y = self.trainDataY
        m = self.trainDataY.shape[0]
        for i in range(epochs):
            yhat = self.predict(self.trainDataX)
            for t in range(theta.shape[1]):
                d_dt = np.add(-yhat,y)
                if self.hasIntercept:
                    if t == 0:
                        theta[:,t] = theta[:,t] - alpha/m*np.sum(d_dt)
                    else:
                        #ex. x has 2 col, theta has 3. thus column relation is x=theta-1 
                        xcol = t-1
                        theta[:,t] = theta[:,t] - alpha/m*np.sum(np.dot(d_dt.T,x))
                else:
                    theta[:,t] = theta[:,t] - alpha/m*np.dot(d_dt.T,x)
        return theta



if __name__ == '__main__':
    input = np.random.rand(100,1)
    output = np.zeros((100,1))
    l = lin_reg(input,output,False)
    y = l.predict(input)
    print(y)
    g = l.gradientDesc(.05,1000)
    print(g)