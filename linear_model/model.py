import numpy as np

class LinearRegression:
    def fit(self, X, y, learning_rate=0.003, iteration= 100000,eps = 0.1):
        self.X = np.array(X)
        self.y = np.array(y)
        self.w = np.random.sample((self.X.shape[1]+1,))
        self.X = np.concatenate((np.ones((self.X.shape[0],1)),self.X), axis=1)
        self.y = self.y.reshape(-1)
        self.learning_rate = learning_rate
        self.eps = eps
        self.iteration = iteration
        
        for i in range(0,self.iteration):
            yhat = self.X.dot(self.w)
            delta = yhat - self.y
            w = self.w - self.learning_rate * (2/self.X.shape[0] * self.X.T.dot(delta))
            if(np.abs(self.w - w).mean() < self.eps):
                break
            self.w = w
        return self

    def predict(self, X):
        X = X.reshape(-1,self.w.shape[0]-1)
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)
        return X.dot(self.w)

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range =  feature_range
        
    def fit(self, X):      
        feature_range = self.feature_range
        
        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)

        data_range = data_max - data_min
        
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        X_scaled = self.scale_ * X + feature_range[0] - data_min * self.scale_
       
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self
        
    def transform(self, X):
        X = np.array(X).astype("float64")
        X *= self.scale_
        X += self.min_
        return X
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
