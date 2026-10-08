import math
from bisect import bisect_right

from scipy.optimize import fsolve  # used for solving implicit equations


def backwardeuler(f1, f2, delta_t, y0):
    '''
    one implicit (backward) Euler step of HumMod's linear differential equation

        dy/dt = f1 - f2 * y

    solved for y1 in  y1 = y0 + delta_t * (f1 - f2 * y1), which is
    unconditionally stable for f2 >= 0:

        y1 = (y0 + delta_t * f1) / (1 + delta_t * f2)

    returns:
    the next y value
    '''
    return (y0 + delta_t * f1) / (1.0 + delta_t * f2)

def diffeq(dydx, delta_t, y0):
    '''
    euler method for estimating the next value of a differential equation
    f is the gradient function for y: dy/dx

    returns the next y value
    '''

    y1 = y0+delta_t*dydx
    return y1

def delay(K, A, B_pre, delta_t):
    '''
    for a given change in time, B steps closer to A. It moves more the further away it is and the greater K is

    B = K * (A - B_init) * delta_t

    '''
    B_post = B_pre + K * (A - B_pre) * delta_t
    return B_post

def stablediffeq(f, delta_t, y0, max_delta_t=None, n=1):
    '''
    currently just a diffeq until I figure out how they're meant to be different
    '''
    return diffeq(f, delta_t, y0)

def impliciteq(f, y_start_estimate, error_limit):
    '''
    Finds y so that
        | y - f(y) | <= Error Limit

    f is the block's residual function: it assigns the block's intermediate
    variables as a side effect and returns the value they imply for y. It is
    always called with a Python float, and once more with the root, so the
    intermediate variables are left consistent with the returned value.
    '''
    def g(y):
        return float(f(float(y)))

    y0 = float(y_start_estimate)
    if abs(y0 - g(y0)) <= error_limit:
        root = y0
    else:
        root = float(fsolve(lambda y: [y[0] - g(y[0])], [y0], xtol=1e-10)[0])
        if not math.isfinite(root) or abs(root - g(root)) > error_limit:
            root = _bisect_fixed_point(g, y0, error_limit)
    g(root)
    return root

def _bisect_fixed_point(g, y0, error_limit):
    '''
    fallback when fsolve does not converge: bracket a sign change of y - g(y)
    around y0 by expanding steps, then bisect it
    '''
    h = lambda y: y - g(y)
    step = max(abs(y0), 1.0) * 0.1
    lo, hi = y0, y0
    for _ in range(60):
        lo, hi = y0 - step, y0 + step
        if h(lo) * h(hi) <= 0:
            break
        step *= 2
    else:
        raise ValueError("impliciteq: no solution found near {}".format(y0))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if abs(h(mid)) <= error_limit or hi - lo < 1e-12:
            return mid
        if h(lo) * h(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def cubic_hermite_spline(x, xarray, yarray, slopes):
    '''
    the curves in HumMod are Cubic Hermite Splines (curves defined by a set of
    x,y coords and a slope at each point). Outside the defined range HumMod
    continues the curve as a straight line with the end point's slope: in the
    HumMod 3.5.2 saved state (Normal.ICS), a curve through (1, 1, slope 0.5) and
    (3, 2, slope 0) evaluates to 0.8055 at x = 0.611, i.e. 1 + 0.5 * (0.611 - 1).

    input:
        xarray and yarray and slopes are lists of the same length, they are the x and y coordinates and associated slopes

    returns a float
    '''
    if x <= xarray[0]:
        return float(yarray[0] + slopes[0] * (x - xarray[0]))
    if x >= xarray[-1]:
        return float(yarray[-1] + slopes[-1] * (x - xarray[-1]))
    i = bisect_right(xarray, x) - 1
    x0, x1 = xarray[i], xarray[i + 1]
    dx = x1 - x0
    t = (x - x0) / dx
    t2 = t * t
    t3 = t2 * t
    return float((2 * t3 - 3 * t2 + 1) * yarray[i]
                 + (t3 - 2 * t2 + t) * dx * slopes[i]
                 + (-2 * t3 + 3 * t2) * yarray[i + 1]
                 + (t3 - t2) * dx * slopes[i + 1])
