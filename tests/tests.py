from stats.stats import (
    cor, lsrl, estimate, residuals, mse, t_conf_linear, se_linear,
    t_stat, p_value
)
from sklearn.linear_model import LinearRegression
import math


def test_corr():
    assert cor([1, 2, 3], [1, 2, 3]) == 1

    assert cor([1, 2, 2, 3], [3, 4, 5, 6], 3) == .949

    assert cor(
        [6, 7, 7, 8, 10, 10, 11, 12, 14, 15, 16],
        [55, 40, 50, 41, 35, 28, 38, 32, 28, 18, 13], 3) == -0.925


def test_lsrl():
    x = [1, 2, 2, 3]
    y = [3, 4, 5, 6]

    model = lsrl(x, y)
    assert isinstance(model, LinearRegression)
    assert round(model.intercept_, 2) == 1.5
    assert round(model.coef_[0], 2) == 1.5

    x = [6, 7, 7, 8, 10, 10, 11, 12, 14, 15, 16]
    y = [55, 40, 50, 41, 35, 28, 38, 32, 28, 18, 13]

    model = lsrl(x, y)

    assert isinstance(model, LinearRegression)
    assert round(model.intercept_, 2) == 70.16
    assert round(model.coef_[0], 2) == -3.39


def test_lsrl_summarize():
    x = [6, 7, 7, 8, 10, 10, 11, 12, 14, 15, 16]
    y = [55, 40, 50, 41, 35, 28, 38, 32, 28, 18, 13]

    assert lsrl(
        x,
        y,
        summarize=True,
        rounding=2) == {'intercept': 70.16, 'slope': -3.39}


def test_estimate():
    intercept = 0.69
    slope = 0.78
    explanatory_var = 3.8

    assert estimate(intercept, slope, explanatory_var) == 3.654


def test_residuals():
    x = [1, 2, 2, 3]
    y = [3, 4, 5, 6]
    model = lsrl(x, y)

    assert residuals(x, y, model, 3) == [0.0, -0.5, 0.5, 0.0]


def test_mse():
    x = [1, 2, 2, 3]
    y = [3, 4, 5, 6]

    model = lsrl(x, y)

    assert mse(x, y, model, rounding=3) == 0.25

    x = [6, 7, 7, 8, 10, 10, 11, 12, 14, 15, 16]
    y = [55, 40, 50, 41, 35, 28, 38, 32, 28, 18, 13]

    model = lsrl(x, y)

    assert round(math.sqrt(mse(x, y, model)), 2) == 5.01


def test_se_linear():
    x = [1, 2, 2, 3]
    mse = 0.25

    assert se_linear(x, mse, 4) == 0.3536


def test_t_conf_linear():
    b1 = 0.7758
    se = 0.1937
    n = 10

    assert t_conf_linear(b1, se, n, rounding=4) == (0.3291, 1.2225)

    b1 = 1.5
    x = [1, 2, 2, 3]
    mse = 0.25
    se = se_linear(x, mse)

    assert t_conf_linear(
        b1,
        se,
        len(x),
        conf_level=.90,
        rounding=2) == (0.47, 2.53)


def test_t_stat():
    x = [1, 2, 2, 3]
    b1 = 1.5
    mse = 0.25
    se = se_linear(x, mse)

    t = t_stat(b1, se, 2)

    assert t == 4.24


def test_p_value():
    x = [1, 2, 2, 3]
    b1 = 1.5
    mse = 0.25
    se = se_linear(x, mse)
    t = t_stat(b1, se, 2)
    k = 1
    df = len(x) - (k+1)

    p = p_value(t, df, 2)

    assert p == .05
