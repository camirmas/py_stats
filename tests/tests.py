from stats.stats import cor, lsrl, estimate, residuals, s_squared
from sklearn.linear_model import LinearRegression


def test_corr():
    assert cor([1, 2, 3], [1, 2, 3]) == 1

    assert cor([1, 2, 2, 3], [3, 4, 5, 6], 3) == .949

    assert cor(
        [6, 7, 7, 8, 10, 10, 11, 12, 14, 15, 16],
        [55, 40, 50, 41, 35, 28, 38, 32, 28, 18, 13], 3) == -0.925


def test_lsrl():
    assert lsrl(
        [1, 2, 2, 3],
        [3, 4, 5, 6],
        rounding=3) == {'intercept': 1.5, 'slope': 1.5}

    x = [6, 7, 7, 8, 10, 10, 11, 12, 14, 15, 16]
    y = [55, 40, 50, 41, 35, 28, 38, 32, 28, 18, 13]

    assert lsrl(x, y, rounding=2) == {'intercept': 70.16, 'slope': -3.39}


def test_lsrl_summarize_false():
    model = lsrl([1, 2, 2, 3], [3, 4, 5, 6], False)

    assert isinstance(model, LinearRegression)


def test_estimate():
    intercept = 0.69
    slope = 0.78
    explanatory_var = 3.8

    assert estimate(intercept, slope, explanatory_var) == 3.654


def test_residuals():
    x = [1, 2, 2, 3]
    y = [3, 4, 5, 6]

    assert residuals(x, y, 3) == [0.0, -0.5, 0.5, 0.0]


def test_s_squared():
    x = [1, 2, 2, 3]
    y = [3, 4, 5, 6]

    assert s_squared(x, y, 3) == 0.25
