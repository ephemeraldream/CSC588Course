import numpy as np
from sklearn.utils import resample
import statsmodels.formula.api as sm
import pandas as pd


class VCMS:

    def __init__(self, DP, m=10, b_1 = 100, b_2 = 100):
        self.b_1 = b_1
        self.b_2 = b_2
        self.DP = DP
        self.m = m


class VCMS_utils:
    @staticmethod
    def get_bootstrap(data: pd.DataFrame, nl):
        D1 = resample(data=data, replace=True, random_state=42, n_samples=nl)
        D2 = resample(data=data, replace=True, random_state=42, n_samples=nl)
        return D1,D2

    @staticmethod
    def fit_models(D1: pd.DataFrame, D2: pd.DataFrame, y_name:str):
        D1_x = D1.loc[:, [y_name]]
        D1_y = D1.drop([y_name])
        D2_x = D2.loc[:, [y_name]]
        D2_y = D2.drop([y_name])
        d1_model = sm.ols(D1_y, D1_x).fit()
        d2_model = sm.ols(D2_y, D1_x).fit()
        return d1_model, d2_model

    @staticmethod
    def standard_error(model, x,y):
        return (model.predict(x) - y)**2


    @staticmethod
    def ab_data_difference(data, y_name, m):
        return np.linspace(start=data[[y_name]].min(), stop=data[[y_name]].max(),num=m)

    @staticmethod
    def get_interval_value(x,y,model, interval):
        for j in interval:
            if model.predict(x,y) <= j:
                return j











def main():
    data =
    utils = VCMS_utils
