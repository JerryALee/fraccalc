import numpy as np


def gamma(x):
    '''
    A better Gamma function

    Parameters
    ----------
    x : float
        Input value of the Gamma function in $\mathbb{R}$.

    Returns
    ----------
    Gamma(x) : float
        For any $x \in \mathbb{Z}^{-}$ or $x = 0$, it will return np.inf.
    '''

    if (x <= 0) and (np.mod(x, 1) == 0):
        return np.inf
    else:
        return np.math.gamma(x)


def gammaRatio(a, b):
    '''
    Calculate Gamma(a) / Gamma(b) in a more general way.

    Parameters
    ----------
    a : float
        The independent variable of the Gamma function of the numerator.
    b : float
        The independent variable of the Gamma function of the denominator.

    Returns
    ----------
    Gamma(a) / Gamma(b) : float
        The return value will not be NaN even if there is any 0 or np.inf in the fraction.
    '''

    if (a <= 0) and (b <= 0) and (np.mod(a, 1) == 0) and (np.mod(b, 1) == 0):
        return (-1)**(a - b) * np.math.factorial(-b) / np.math.factorial(-a)
    else:
        return gamma(a) / gamma(b)