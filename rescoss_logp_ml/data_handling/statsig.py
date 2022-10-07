"""
From https://github.com/jensengroup/statsig/blob/master/statsig.py (Accessed 25.04.22)

The license from the original repo is reproduced here as requested: 

MIT License

Copyright (c) 2016 Jensen Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np


def correl(X, Y):
    (N,) = X.shape

    if N < 9:
        print(f"not enough points. {N} datapoints given. at least 9 is required")
        return

    r = np.corrcoef(X, Y)[0][1]
    r_sig = 1.96 / np.sqrt(N - 2 + 1.96 ** 2)
    F_plus = 0.5 * np.log((1 + r) / (1 - r)) + r_sig
    F_minus = 0.5 * np.log((1 + r) / (1 - r)) - r_sig
    le = r - (np.exp(2 * F_minus) - 1) / (np.exp(2 * F_minus) + 1)
    ue = (np.exp(2 * F_plus) - 1) / (np.exp(2 * F_plus) + 1) - r

    return r, le, ue


def rmse(X, Y):
    """
    Root-Mean-Square Error
    Lower Error = RMSE \left( 1- \sqrt{ 1- \frac{1.96\sqrt{2}}{\sqrt{N-1}} }  \right )
    Upper Error = RMSE \left(    \sqrt{ 1+ \frac{1.96\sqrt{2}}{\sqrt{N-1}} } - 1 \right )
    This only works for N >= 8.6832, otherwise the lower error will be
    imaginary.
    Parameters:
    X -- One dimensional Numpy array of floats
    Y -- One dimensional Numpy array of floats
    Returns:
    rmse -- Root-mean-square error between X and Y
    le -- Lower error on the RMSE value
    ue -- Upper error on the RMSE value
    """

    (N,) = X.shape

    if N < 9:
        print(f"Not enough points. {N} datapoints given. At least 9 is required")
        return

    diff = X - Y
    diff = diff ** 2
    rmse = np.sqrt(diff.mean())

    le = rmse * (1.0 - np.sqrt(1 - 1.96 * np.sqrt(2.0) / np.sqrt(N - 1)))
    ue = rmse * (np.sqrt(1 + 1.96 * np.sqrt(2.0) / np.sqrt(N - 1)) - 1)

    return rmse, le, ue


def mae(X, Y):
    """
    Mean Absolute Error (MAE)
    Lower Error =  MAE_X \left( 1- \sqrt{ 1- \frac{1.96\sqrt{2}}{\sqrt{N-1}} }  \right )
    Upper Error =  MAE_X \left(  \sqrt{ 1+ \frac{1.96\sqrt{2}}{\sqrt{N-1}} }-1  \right )
    Parameters:
    X -- One dimensional Numpy array of floats
    Y -- One dimensional Numpy array of floats
    Returns:
    mae -- Mean-absolute error between X and Y
    le -- Lower error on the MAE value
    ue -- Upper error on the MAE value
    """

    (N,) = X.shape

    mae = np.abs(X - Y)
    mae = mae.mean()

    le = mae * (1 - np.sqrt(1 - 1.96 * np.sqrt(2) / np.sqrt(N - 1)))
    ue = mae * (np.sqrt(1 + 1.96 * np.sqrt(2) / np.sqrt(N - 1)) - 1)

    return mae, le, ue
