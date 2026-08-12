# -*- coding: utf-8 -*-
"""
some test problems from regu

@author: Admin
"""
import numpy as np
import scipy

def gravity_model(n,a,b,d):
    """ Defines the square matrix A by mid-point sampling
    
    Parameters:
    ------------
    n : int
        order of the model
    a : float
        bottom limit of the Fredholm integral
    b : float
        upper limit of the Fredholm integral
    d : float
        vertical distance from the source line (x') to measurement line (x)

    Returns:
    ------------
    A : numpy ndArray
        n x n real sensing matrix
    """
    dx = 1/n
    dxl = (b-a)/n
    x = dx * (np.arange(n) + 0.5) 
    xl = a + dxl * (np.arange(n) + 0.5) 
    X,XL = np.meshgrid(x,xl)
    A = dx * (d/(d**2 + (X-XL)**2)**(3/2)) 
    return A, x

def density_piecewise(n):
    """ Piecewise source function
    
    Defines a function of order n for which the first 1/3 of the source points are 2, and the rest are 1.
    
    Parameters:
    ------------
    n : int
        order of the model

    Returns:
    ------------
    rhox : numpy 1dArray
        pieciwise source term
    """
    nn = int(n/3)
    rhox = np.ones(n)
    rhox[:nn] = 2
    return rhox

def density_sin(x, fp):
    """ Sinusoidal source function
    
    Defines a function or order len(x) for a sinusoidal source function.
    
    Parameters:
    -----------
    x : 1dArray
        measured coordinates
    fp : float
        spatial frequency of the phenomenon
    Returns:
    ------------
    rhox : numpy 1dArray
        sinusoidal source term
    """
    rhox = np.sin(2*np.pi*fp*x)
    return rhox

def density_rechann(x, hann_x = 0.5):
    """ Rectangular + hanning density function
    
    Parameters:
    ------------
    n : int
        order of the model

    Returns:
    ------------
    rhox : numpy 1dArray
        pieciwise source term
    """
    n = len(x)
    nh = len(x[x>hann_x])
    hw = scipy.signal.windows.hann(2*nh)
    rhox = 2*np.ones(n)
    rhox[n-nh:] = hw[nh:]+1
    return rhox

