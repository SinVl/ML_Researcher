import numpy as np

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