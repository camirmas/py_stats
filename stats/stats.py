"""
This package provides convenience functions for basic statistics calculations.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from math import sqrt


# Bivariate Data Analysis Functions

def cor(x, y, rounding=None):
    """Returns the correlation coefficient between two collections."""
    x = pd.Series(x)
    y = pd.Series(y)
    r = x.corr(y)

    if rounding:
        return round(r, rounding)
    return r


def scatter_plot(x, y, x_label='x', y_label='y', zero_line=False):
    """Returns a basic scatterplot for two collections."""
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if zero_line:
        plt.axhline(linestyle="--")

    return plt


def lsrl(x, y, summarize=False, rounding=None):
    """
    Calculates the least squares regression line (LSRL) for two collections.
    Returns either a simple summary, or the model object itself.
    """
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    model = LinearRegression().fit(x, y)

    if summarize:
        intercept = model.intercept_
        slope = model.coef_[0]

        if rounding:
            intercept = round(intercept, rounding)
            slope = round(slope, rounding)

        return {"intercept": intercept, "slope": slope}

    return model


def regression_plot(x, y, model, x_label='x', y_label='y'):
    """Returns a basic regression line + scatter plot for two collections."""
    plt = scatter_plot(x, y, x_label, y_label)

    x = np.array(x).reshape((-1, 1))
    plt.plot(x, model.predict(x), color='blue')

    return plt


def estimate(intercept, slope, explanatory_var):
    """
    Returns the expected value (response) for a given explanatory variable.
    """
    return intercept + slope * explanatory_var


def residuals(x, y, model, rounding=None):
    """
    Returns the residual (observed - expected) for a simple linear regression.
    """
    result = []

    for i, _ in enumerate(x):
        res = y[i] - model.predict([[x[i]]])[0]

        if rounding:
            result.append(round(res, rounding))
        else:
            result.append(res)

    return result


def residual_plot(x, y, model):
    """Returns a residual plot for a regression."""
    return scatter_plot(x, residuals(x, y, model), zero_line=True)


def mse(x, y, model, k=1, rounding=None):
    """Returns the mean squared error (s^2) for a regression."""
    sse = 0

    n = len(x)

    for i, _ in enumerate(x):
        res = (y[i] - model.predict([[x[i]]])[0]) ** 2
        sse += res

    result = sse / (n - (k + 1))

    if rounding:
        return round(result, rounding)

    return result


def se_linear(x, mse, rounding=None):
    s = sqrt(mse)
    sum_squares = 0
    mean_x = np.mean(x)

    for val in x:
        sum_squares += (val - mean_x) ** 2

    result = s / sqrt(sum_squares)

    if rounding:
        result = round(result, rounding)

    return result


def t_conf_linear(b1, se, n, k=1, conf_level=.95, rounding=None):
    df = n - (k + 1)
    t_crit = stats.t.ppf(conf_level + (1-conf_level)/2, df=df)

    lower = b1 - t_crit * se
    upper = b1 + t_crit * se

    if rounding:
        lower = round(lower, rounding)
        upper = round(upper, rounding)

    return (lower, upper)


def t_stat(b1, se, rounding=None):
    """Calculates the t stat given the slope and standard error."""
    res = b1 / se

    if rounding:
        res = round(res, rounding)

    return res


def p_value(t, df, rounding=None):
    """Calculates the p-value given a t stat and degrees of freedom."""
    res = 2*(1 - stats.t.cdf(abs(t), df))

    if rounding:
        res = round(res, rounding)

    return res


def run_regression(x, y, conf_level=.95):
    """Runs a simple linear regression analysis."""
    model = lsrl(x, y)
    s_2 = mse(x, y, model)
    se = se_linear(x, s_2)
    conf = t_conf_linear(model.coef_[0], se, len(x), conf_level=conf_level)
    t = t_stat(model.coef_[0], se)
    k = 1
    df = len(x) - (k + 1)
    p = p_value(t, df)

    return {
        "model": model,
        "intercept": model.intercept_,
        "slope": model.coef_[0],
        "MSE": s_2,
        "SE": se,
        "t_stat": t,
        "p_value": p,
        "conf_level": conf_level,
        "conf_interval": conf,
        "df": df,
        "regression_plot": lambda: regression_plot(x, y, model).show(),
        "residual_plot": lambda: residual_plot(x, y, model).show()
    }
