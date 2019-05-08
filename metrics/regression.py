import numpy as np

def mean_absolute_error(y_true,y_pred):
    return np.abs(y_true - y_pred).mean()

def mean_squared_error(y_true,y_pred):
	return ((y_true - y_pred)**2).mean()

def mean_squared_log_error(y_true,y_pred):
	return mean_squared_error(np.log1p(y_true),np.log1p(y_pred))

def root_mean_squared_log_error(y_true,y_pred):
    return mean_squared_log_error(y_true,y_pred)**0.5