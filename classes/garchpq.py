import numpy as np
from scipy.optimize import minimize
from arch import arch_model
import matplotlib.pyplot as plt
import pandas as pd


class GARCH:
    def __init__(self, p=1, o=0, q=1):
        self.p = p
        self.q = q
        # differentiation parameter
        self.o = o
        # ini conditional variance
        self.sigma2 = None
        self.resid = None

    def log_likelihood(self, parameters, data):
        """
        Compute LL
        Followed and generalised the methodology of Kevin Sheppard:
        https://www.kevinsheppard.com/teaching/python/notes/notebooks/example-gjr-garch/
        """
        # Define Parameters
        mu = parameters[0]
        omega = parameters[1]
        alpha = parameters[2: self.p + 2]
        beta = parameters[self.p + 2: -1] if self.o == 1 else parameters[self.p + 2:]
        gamma = parameters[-1] if self.o == 1 else 0

        eps = data - mu
        # Initialize the variance and residual
        sigma2 = np.zeros_like(data)
        resid = np.zeros_like(data)
        # Initialize the log-likelihood
        LL = 0

        # Iterate over the data and calculate the variance and log-likelihood
        start = max(self.p, self.q)
        sigma2[start - 1] = omega / (1 - np.sum(alpha) - np.sum(beta))
        for t in range(start, len(data)):
            resid[t] = eps[t] - np.sum(alpha * eps[t - self.p: t])
            sigma2[t] = (
                    omega
                    + np.sum(
                alpha * eps[t - self.p: t] ** 2 + beta * sigma2[t - self.p: t]
            )
                    + (eps[t - 1] < 0) * gamma * eps[t - 1] ** 2
            )

            if self.o == 0:
                LL += -np.log(np.sqrt(2 * np.pi * sigma2[t])) - 0.5 * (
                        data[t] ** 2 / sigma2[t]
                )
            else:
                LL += -np.log(np.sqrt(2 * np.pi * sigma2[t])) - 0.5 * (
                        data[t] ** 2 / sigma2[t]
                )

        # squared of conditional variance
        self.sigma2 = sigma2
        self.resid = resid
        return -LL

    def fit(self, data, initial):
        """
        Return fitted value
        Last value is the gamma parameter
        First 2 values are the mean and omega
        """
        # constraint on params
        if self.o == 1:
            cons = {"type": "ineq", "fun": lambda x: 1 - sum(x[2:-1]) - x[-1] / 2}
        else:
            cons = {"type": "ineq", "fun": lambda x: 1 - sum(x[2:])}
        # Use the minimize function to find the parameters that maximize the log-likelihood
        finfo = np.finfo(np.float32)
        a = [
            (-10 * abs(data.mean()), 10 * abs(data.mean())),
            (finfo.eps, 2 * data.var()),
        ]
        b = [(0.0, 1.0)] * (self.p + self.q + self.o)
        bo = a + b
        param_opt = minimize(
            self.log_likelihood,
            initial,
            args=(data,),
            constraints=cons,
            method="SLSQP",
            bounds=bo,
        )

        # Extract the GARCH and ARCH parameters
        return param_opt

    def predict(self, X, parameters, error=False):
        """
        Predict using previous fitted parameters
        """
        mu = parameters[0]
        if not error:
            err = X - mu
        else:
            err = X
        omega = parameters[1]
        alpha = parameters[2: self.p + 2]
        beta = parameters[self.p + 2: -1] if self.o == 1 else parameters[self.p + 2:]
        gamma = parameters[-1] if self.o == 1 else 0

        start = max(self.p, self.q)
        sigma2 = np.zeros(len(err))
        # err = np.zeros(horizon)
        # err[0] = self.resid[-start]
        sigma2[0] = self.sigma2[-start]

        for t in range(1, len(err)):
            sigma2[t] = (
                    omega
                    + np.sum(
                alpha * err[t - self.p: t] ** 2 + beta * sigma2[t - self.p: t]
            )
                    + (err[t - 1] < 0) * gamma * err[t - 1] ** 2
            )

        return np.sqrt(sigma2)


class Estimation:
    def __init__(self):
        self.output = None
        self.conditional_vol_train = None
        self.cond_vol_pred = None
        self.params_opt = None

    def fit_and_predictions(self, X_train, X_test, init_params):
        self.output = np.zeros_like(X_test)
        garch = GARCH(p=1, q=1, o=0)
        self.params_opt = garch.fit(X_train, init_params).x
        self.conditional_vol_train = np.sqrt(garch.sigma2)
        self.cond_vol_pred = pd.Series(garch.predict(X_test, self.params_opt, error=True))

    def plot_pred(self, X_test):
        true_volatility = X_test.rolling(5).std()
        self.cond_vol_pred.index = true_volatility.index
        plt.plot(self.cond_vol_pred, label='conditional volatility prediction')
        plt.plot(true_volatility, label='true volatility (rolling on 5 days)')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.title('true volatility vs predicted under garch(1,1)')


if __name__ == "__main__":
    data = pd.read_csv('data/dataset.csv').iloc[:, 1].dropna()
    X_train, X_test = data[:int(len(data) * 0.75)], data[int(len(data) * 0.75):]
    serie = X_train.copy().values * 10
    y = X_test.copy().values * 10

    params = [serie.mean(), serie.var() * 0.01, 0.05, 0.8]

    X_train, X_test = serie[:int(len(serie) * 0.8)], serie[int(len(serie) * 0.8):]
    est = Estimation()
    X = pd.Series(X_train)
    y = pd.Series(X_test)
    est.fit_and_predictions(X, y, params)
    est.plot_pred(y)

    am = arch_model(X, p=1, q=1)
    res = am.fit()
    us = pd.Series(est.conditional_vol_train)
    lib = res.conditional_volatility
    us.index = lib.index
    plt.figure(2)
    plt.plot(lib)
    plt.plot(us)
