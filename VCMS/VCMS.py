import bisect
import math

import numpy as np
from sklearn.utils import resample
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from scipy import optimize
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
        D1 = resample(data, replace=True, n_samples=nl)
        D2 = resample(data, replace=True, n_samples=nl)
        return D1,D2

    @staticmethod
    def fit_models(D1: pd.DataFrame, D2: pd.DataFrame, y_name:str, columns):
        formula = f'{y_name} ~'
        for i in columns:
            if i != y_name: formula += f'{i} + '
        formula = formula[:-2]
        d1_model = sm.ols(formula=formula, data=D1)
        results_1 = d1_model.fit()
        d2_model = sm.ols(formula=formula, data=D2)
        results_2 = d2_model.fit()

        return results_1, results_2

    @staticmethod
    def standard_error(model, x,y):
        return (model.predict(x).iloc[0].item() - y)**2


    @staticmethod
    def get_interval(data, y_name, m):
        return np.linspace(0, stop=data[[y_name]].max(),num=m)

    @staticmethod
    def get_interval_value(x: pd.Series,y,model, interval):
        errors = VCMS_utils.standard_error(model, x, y)
        return bisect.bisect(interval, errors)-1


    @staticmethod
    def get_Ns(model, data, y_name, interval, m):
        N_s = np.zeros(shape=(m,))
        #interval = VCMS_utils.get_interval(data, y_name, m)
        for i in range(data.shape[0]):
            x = data.iloc[i].drop(y_name)
            y = data.iloc[i,:][y_name]
            N_s[VCMS_utils.get_interval_value(x,y, model, interval)] += 1

        return N_s


    @staticmethod
    def step_1_7(data, y_name, m, b_1, nl, transfer: dict):
        risks_b_1 = np.zeros(shape=(b_1, m))
        for i in range(b_1):
            D1, D2 = VCMS_utils.get_bootstrap(data, nl)
            model_d1, model_d2 = VCMS_utils.fit_models(D1, D2, 'medv', columns=transfer['columns'])
            interval = VCMS_utils.get_interval(data, 'medv', m)
            N_s_d2 = VCMS_utils.get_Ns(model_d1, D2, y_name, interval, m)
            N_s_d1 = VCMS_utils.get_Ns(model_d2, D1, y_name, interval, m)
            interval = (interval / D1.shape[0]).reshape((m,))
            N_s_d2 = np.multiply(N_s_d2,interval)
            N_s_d1 = np.multiply(N_s_d1,interval)
            risk_m = np.abs(N_s_d1 - N_s_d2)
            risks_b_1[i] = risk_m
        resulted_mean = sum(risks_b_1.mean(axis=0))
        return resulted_mean



    @staticmethod
    def step_8(data, y_name, m, b_1, b_2, nl, transfer: dict):
        means_b_2 = np.zeros(shape=(b_2, ))
        for i in range(b_2):
            means_b_2[i] = VCMS_utils.step_1_7(data, y_name, m, b_1, nl, transfer)
        print(f"SUB STEP OF {nl} of {np.mean(means_b_2)} IS DONE")
        return np.mean(means_b_2)


    @staticmethod
    def obj(x, list_of_xi, nls, c):
        x = x[0]
        return np.sum((list_of_xi - c*np.sqrt((x/nls)*np.log((2*nls*math.exp(1))/x)))**2)

    @staticmethod
    def obj_f_minimization(list_of_xi: np.array, nls: np.array, c):
        res = optimize.minimize(VCMS_utils.obj, args=(list_of_xi,nls, c), x0=np.array([1]), method='nelder-mead')
        return res['x'][0]

    @staticmethod
    def step_9(data, nls, b, transfer):
        results_of_nls = []
        for i in nls:
            results_of_nls.append(VCMS_utils.step_8(data,'medv', 10, b, b, i, transfer))
        #print(results_of_nls)
        return np.array(results_of_nls)

    @staticmethod
    def optimize_over_d_vars(data: pd.DataFrame):
        cols = data.columns
        transfer = {}
        VCD_d = {}
        nls = np.linspace(50, data.shape[0], 5, dtype=np.int16)
        for i in range(len(cols)-1):
            transfer['columns'] = cols[:i+1]
            results_of_nl = VCMS_utils.step_9(data, nls, 5, transfer)
            VCD_d[str(i)] = VCMS_utils.obj_f_minimization(results_of_nl, nls, 10)
            print(f"THE VC DIMENSION FOR COLUMNS {transfer['columns']} IS {VCMS_utils.obj_f_minimization(results_of_nl, nls, 10)}")
            print(f"######## STEP {i} IS COMPLETED. WORKING ON THE NEXT")
        print(f"THE RESULT IS: {VCD_d}")
        np.save('bostonVCD.npy', VCD_d)













def main():
    data = pd.read_csv("Boston.csv")
    # utils = VCMS_utils
    # D1,D2 = utils.get_bootstrap(data, 50)
    # d1_model, d2_model = utils.fit_models(D1, D2, 'medv')
    # interval = VCMS_utils.get_interval(D1, 'medv', 50)
    # #utils.get_Ns(d1_model, D1, 'medv', interval,100)
    # VCMS_utils.step_8(D1,D2, d1_model, d2_model,'medv', interval, 50, 5, 5)
    nls = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    xis = [14.930222222222222, 15.559566666666665, 11.680474074074073, 9.848819444444445, 8.351368888888889,
     7.551914814814813, 7.1897730158730155, 7.060258333333333, 6.535266666666667]

    #xis = VCMS_utils.step_9(data, nls, 10)
    #print(VCMS_utils.obj(5, nls, xis, 5))
    #results = []
    #for i in range(10):
    #    results.append(VCMS_utils.obj_f_minimization(xis, nls, 13, len(nls)))
    #VCMS_utils.step_9(data, nls)
    #print(results)
    VCMS_utils.optimize_over_d_vars(data)







if __name__ == "__main__":
    main()