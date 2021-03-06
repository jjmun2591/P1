import numpy as np
np.random.seed(1)

class lin_reg():
    
    def __init__(self,trainDataX,trainDataY):
        self.theta = np.random.rand(1,input.shape[1])
        self.trainDataX = trainDataX
        self.trainDataY = trainDataY

    def predict(self,X,theta):
        y = np.dot(X,theta.T)
        return y
    
    def cost(self,x,theta):
        y = self.trainDataY
        yhat = self.predict(x,theta)
        m = y.shape[1]
        XO_y = yhat-y
        return 1/2/m*np.dot(XO_y.T,XO_y)

    def gradientDesc(self,alpha,epochs):
        theta = self.theta
        x = self.trainDataX
        y = self.trainDataY
        m = self.trainDataY.shape[0]
        for i in range(epochs):
            yhat = self.predict(x,theta)
            
            d_dt = np.add(yhat,-y) #nx1
            theta = theta - alpha/m*np.dot(d_dt.T,x)#2xn nx1
            cost = self.cost(x,theta)
            #print(cost)

        return theta
    
    def closedSolution(self):
        x = self.trainDataX
        y = self.trainDataY

        return np.dot(np.linalg.inv(np.dot(x.T,x)),np.dot(x.T,y))



if __name__ == '__main__':
    input = np.array([[1,2,3,4,5,6,7,8,9],[1,1,1,1,1,1,1,1,1]]).T
    output = np.array([[2,4,6,8,10,12,14,16,18]]).T
    l = lin_reg(input,output)
    g = l.gradientDesc(.05,10000)
    print(g)
    print(l.closedSolution())