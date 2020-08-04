"""
This package provides convenience functions for basic statistics calculations.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


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


def lsrl(x, y, summarize=True, rounding=None):
    """
    Calculates the least squares regression line (LSRL) for two collections.
    Returns either a simple summary, or the model object itself.
    """
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    model = LinearRegression().fit(x, y)

    if not summarize:
        return model

    intercept = model.intercept_
    slope = model.coef_[0]

    if rounding:
        intercept = round(intercept, rounding)
        slope = round(slope, rounding)

    return {"intercept": intercept, "slope": slope}


def regression_plot(x, y, x_label='x', y_label='y'):
    """Returns a basic regression line + scatter plot for two collections."""
    plt = scatter_plot(x, y, x_label, y_label)
    model = lsrl(x, y, False)

    x = np.array(x).reshape((-1, 1))
    plt.plot(x, model.predict(x), color='blue')

    return plt


def estimate(intercept, slope, explanatory_var):
    """
    Returns the expected value (response) for a given explanatory variable.
    """
    return intercept + slope * explanatory_var


def residuals(x, y, rounding=None):
    """
    Returns the residual (observed - expected) for a simple linear regression.
    """
    model = lsrl(x, y, summarize=False)
    result = []

    for i, _ in enumerate(x):
        res = y[i] - model.predict([[x[i]]])[0]

        if rounding:
            result.append(round(res, rounding))
        else:
            result.append(res)

    return result


def residual_plot(x, y):
    """Returns a residual plot for a regression."""
    return scatter_plot(x, residuals(x, y), zero_line=True)


def mse(x, y, rounding=None, k=1):
    """Returns the mean squared error (s^2) for a regression."""
    model = lsrl(x, y, summarize=False)
    sse = 0

    n = len(x)

    for i, _ in enumerate(x):
        res = (y[i] - model.predict([[x[i]]])[0]) ** 2
        sse += res

    result = sse / (n - (k + 1))

    if rounding:
        return round(result, rounding)

    return result
