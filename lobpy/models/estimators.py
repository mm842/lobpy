"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller

estimators.py

Defines estimators which could be used for calibration of the models.

"""



import math
import numpy as np
import scipy.optimize as sopt
import warnings


class CalibrationWarning(RuntimeWarning):
    pass;


def estimate_recgamma_diff_old(data, dt):
    """ Returns the parameters according to the the estimators (nu, mu, sigma) of the mean reverting diffusion process
    based on observations data observed at the uniform time grid with time steps of size dt. 
    See Leonenko, Suvak (2010) equations (5.5), (6.1) and (6.2) for the estimators
    ----------
    args:
        data:    data on equidistant time grid
        dt:      distance between two time points in the data
    output:
        mu:    estimator for mean reversion level
        nu:    estimator for mean reversion speed
        sigma: estimator for volatility
    """

    try:
        N = int(data.size) - 1
    except :  ##ValueError:
        warnings.warn("input argument data was not an ndarray")
        return None

    # first, we reparametrize to the classical reciprocal gamma diffusion framework: mu = alpha / (beta - 1), sigma ** 2 = 2 nu /(beta-1)
    
    # See Section 6, Leonenko, Suvak (2010)    
    bm1 = np.mean(data[1::])
    bm2 = np.mean(np.power(data[1::],2))

    alphahat = bm1 * bm2 / (bm2 - (bm1 ** 2))
    bethat = 1. + bm2  / (bm2 - (bm1 ** 2))
        
        # See Section 5, Leonenko, Suvak (2010)
    bessel1 = lambda x: -(bethat - 1.) * x + alphahat
    muhat = alphahat/ (bethat -1.)

    
    
    sum2 = np.sum(bessel1(data[:N:]) / np.power(data[:N:],2) * (data[1::] - muhat))
    sum1 = np.sum(bessel1(data[:N:]) / np.power(data[:N:],2) * (data[:N:] - muhat))

    frac = sum1 / float(sum2)
    
    if frac <= 0:
        warnings.warn("Calibration yields infinite mean reversion speed", CalibrationWarning)
        tnu = float('nan')
    else:    
        tnu = math.log(frac) / dt

    sigma2 = 2*tnu / (bethat - 1.)

    if sigma2 < 0:
        warnings.warn("Calibration yields negative volatility.\n Esimators sigma2 = 2(nu) / (beta - 1) = {0}, nu = {1}, beta = {2}\n Set sigma := 0".format(sigma2, tnu, bethat), CalibrationWarning)
        sigma2 = 0


    tnu = np.log(sum1 / sum2) / dt
            
    return muhat, tnu, (math.sqrt(sigma2))


def estimate_recgamma_diff(data, dt):
    """ Returns the parameters according to the the estimators (nu, mu, sigma) of the mean reverting diffusion process
    based on observations data observed at the uniform time grid with time steps of size dt. 
    See Leonenko, Suvak (2010) equations (5.5), (6.1) and (6.2) for the estimators:
    We parametrize the process as  
    d Xt = nu(mu - Xt) + sqrt(2 nu (Xt ** 2) / c ) dWt 
    for alpha, nu > 0, beta > 1 (i.e. mu = alpha / (beta-1), sigma = sqrt(2 a nu)
    ----------
    args:
        data:    data on equidistant time grid
        dt:      distance between two time points in the data
    output:
        mu:    estimator for mean reversion level
    nu:    estimator for mean reversion speed
        sigma: estimator for volatility
    """
    
    try:
        N = int(data.size) - 1
    except :  ##ValueError:
        warnings.warn("input argument data was not an ndarray")
        return None
    
    # first, we reparametrize to the classical reciprocal gamma diffusion framework: mu = alpha / (beta - 1), sigma ** 2 = 2 nu /(beta-1)
    
    # See Section 6, Leonenko, Suvak (2010)    
    est_mu = np.mean(data[1::])    
    bm2 = np.mean(np.square(data[1::]))
    var_data = (bm2 - (est_mu ** 2))
    # Check if Var(data) == 0)
    if var_data == 0:
        warnings.warn("Input data seem to be constant in time: sum data ** 2 - (sum data) ** 2 == 0.", CalibrationWarning)
        return est_mu, 0, 0

    #alphahat = bm1 * bm2 / var_data
    # bethat = \hat \beta - 1. (in the reference paper)
    est_c = bm2  / var_data
    
    
    sum1 = np.sum(np.true_divide(np.square((data[:N:] - est_mu)), np.square(data[:N:])))
    sum2 = np.sum(np.multiply(np.true_divide((data[:N:] - est_mu), np.square(data[:N:])), (data[1::] - est_mu)))

    tnu = 0
    
    if sum2 <= 0:
        warnings.warn("Calibration yields infinite mean reversion speed", CalibrationWarning)
        tnu = float('nan')
    else:    
        tnu = math.log(sum1 / sum2) / dt

    sigma2 = 2 * tnu / est_c

    if sigma2 < 0:
        warnings.warn("Calibration yields negative volatility.\n Esimators sigma**2 = 2(nu) / beta = {0}, nu = {1}, beta = {2}\n Set sigma := 0".format(sigma2, tnu, est_c), CalibrationWarning)
        sigma2 = 0
    #mu = alphahat/ (bethat -1.) = bm1
    #sigma = sqrt(2tnu / (betahat - 1))
    return est_mu, tnu, math.sqrt(sigma2)


def _realized_var(data):
    ''' compute realized variance of data '''
    res = np.subtract(data[1::], data[:-1:])
    return np.sum(np.power(res, 2))


def _realized_covar(data1, data2):
    """ compute realized variance of data1, data2 and the realized covariance. Output: RV(data1), RV(data2), RCV(data1,data2) """
    res1 = np.subtract(data1[1::], data1[:-1:])
    res2 = np.subtract(data2[1::], data2[:-1:])
    realized_covar = np.sum(np.multiply(res1, res2)) 
    realized_var1 = np.sum(np.power(res1, 2)) 
    realized_var2 = np.sum(np.power(res2, 2))    
    return realized_var1, realized_var2, realized_covar


def estimate_vol_rv(data, t):
    """ Estimates the quadratic correlation of data1 and data2, assuming these are time series on a uniform grid with mesh size dt, of processes with quadratic variation
    More data1 and data2 are assumed to follow 
    d[X](t) = sigma^2 dt, 

    The estimation uses realized variation
    
    ----------
    args:
    data
    t

    output:
    sigma
    """
    
    return math.sqrt(_realized_var(data) / float(t))



def estimate_vol_2d_rv_incr(data1, data2, time_incr=0.1, log=True):
    """ Estimates the quadratic correlation of data1 and data2, assuming these are time series on a uniform partition of [0,t], of processes with quadratic variation
    More data1 and data2 are assumed to follow 
    d[X1](t) = sigma1^2 X1(t)^2 dt, 
    d[X2](t) = sigma2^2 X2(t)^2 dt, 
    d[X1,X2](t) = sigma1 sigma2 X1(t) X2(t) rho dt

    If log=False, then 
    d[X1](t) = sigma1^2  dt, 
    d[X2](t) = sigma2^2  dt, 
    d[X1,X2](t) = sigma1 sigma2  rho dt    

    The estimation uses realized covariation on logX1 and logX2    

    ----------
    args:
    data1    data array for X1
    data2    data array for X2
    time_incr        time increment
    log=True     if True, then estimation based on log of data1 and data2, else in plain format.

    output:
    sigma_1, sigma_2, rho

    """

    length = len(data1)
    if not (length==len(data2)):
        print("Error: data1 and data2 have not the same length.")
        raise Exception
    t = (length-1) * float(time_incr)
    if log:
        rvb, rva, rcv = _realized_covar(np.log(data1), np.log(data2))
    else:
        rvb, rva, rcv = _realized_covar(data1, data2)
    return (math.sqrt(float(rvb) / float(t))) , (math.sqrt(float(rva) / float(t))), float(rcv) / float(math.sqrt(rvb * rva))

def estimate_vol_2d_rv(data1, data2, t=1., log=True):
    """ Estimates the quadratic correlation of data1 and data2, assuming these are time series on a uniform partition of [0,t], of processes with quadratic variation
    More data1 and data2 are assumed to follow 
    d[X1](t) = sigma1^2 X1(t)^2 dt, 
    d[X2](t) = sigma2^2 X2(t)^2 dt, 
    d[X1,X2](t) = sigma1 sigma2 X1(t) X2(t) rho dt

    If log=False, then 
    d[X1](t) = sigma1^2  dt, 
    d[X2](t) = sigma2^2  dt, 
    d[X1,X2](t) = sigma1 sigma2  rho dt    

    The estimation uses realized covariation on logX1 and logX2    

    ----------
    args:
    data1    data array for X1
    data2    data array for X2
    t        terminal time


    output:
    sigma_1, sigma_2, rho

    """

    if log:
        rvb, rva, rcv = _realized_covar(np.log(data1), np.log(data2))
    else:
        rvb, rva, rcv = _realized_covar(data1, data2)
    return (math.sqrt(float(rvb) / float(t))) , (math.sqrt(float(rva) / float(t))), float(rcv) / float(math.sqrt(rvb * rva))


def estimate_vol_gBM(data1, data2, time_incr=0.1):
    """ Estimate vol and correlation of two geometric Brownian motion samples with time samples on a grid with mesh size time_incr using estimate_vol_2d_rv_incr, the drift parameter and mean rev paramters are set to 0. 

    ----------
    args:
    data1    data array for X1
    data2    data array for X2
    time_incr        time increment
    log=True     if True, then estimation based on log of data1 and data2, else in plain format.

    output:
    [0, 0, sigma_1], [0,0, sigma_2], rho    format to be used direclty in a LOBLinear model object
    """

    sigma_bid, sigma_ask, rho = estimate_vol_2d_rv_incr(data1, data2, time_incr, log=True)    
    return [float(0), float(0), sigma_bid], [float(0), float(0), sigma_ask], rho



def estimate_log_corr_rv(data1, data2):
    """ Estimates the quadratic correlation of data1 and data2, assuming these are time series on a uniform partition of [0,t], of processes with quadratic variation
    More data1 and data2 are assumed to follow 
    d[X1](t) = sigma1^2 X1(t)^2 dt, 
    d[X2](t) = sigma2^2 X2(t)^2 dt, 
    d[X1,X2](t) = sigma1 sigma2 X1(t) X2(t) rho

    The estimation uses realized covariation on logX1 and logX2
    
    ----------
    args:
    data1    data array for X1
    data2    data array for X2
    t        terminal time


    output:
    rho

    """
    __, __, rho = estimate_vol_2d_rv(data1, data2, 1., log=True)
    return rho

def realized_covar_general(time_arr, data_bid_arr, data_ask_arr, time_incr, start_time=34200., end_time=57600.):
    ''' Estimates the variance and covariance of log of bid and ask side by computing realized (co)variation of log of bid and ask data on [start_time, end_time] partitioned in small intervals of length time_incr. More precisely, the model returns estimators for (sigma_b^2, sigma_a^2, sigma_b^2 rho) under the assumption that
    d [log(data_bid)]_t = sigma_b^2 t
    d [log(data_ask)]_t = sigma_a^2 t
    d [log(data_ask), log(data_bid)]_t = rho t
    
    ----------
    Args:
    time_arr      array with time steps corresponding to data
    data_bid_arr  array with bid data
    data_ask_arr  array with ask data
    time_incr     mesh size of the time grid for computation
    start_time    time point to start calculation
    end_time      time point to end calculation

    Output:
    rcv           covariance estimator
    rv_bid        realized variation on bid side
    rv_ask        realized variation on ask side
    '''

    if len(data_bid_arr) != len(data_ask_arr):
        print("Invalid size of arrays")
        return 0,0,0
    if len(time_arr) != len(data_ask_arr):
        print("Invalid size of arrays")
        return 0,0,0

    if end_time > time_arr[-1]:
        end_time = time_arr[-1]

    if start_time < time_arr[0]:
        start_time = time_arr[0]
        
    data_bid_last = 0.
    data_ask_last = 0.
    timelat = 0.
    rv_bid = 0.
    rv_ask = 0.
    rcv_bidask = 0.
    ctr_data = 0
    
    for data_bid, data_ask, time in zip(reversed(data_bid_arr), reversed(data_ask_arr), reversed(time_arr)):
        logdata_bid = np.log(data_bid)
        logdata_ask = np.log(data_ask)                

        if time > end_time:
            continue

        if ctr_data == 0:
            # Initialize values
            data_bid_last = data_bid
            data_ask_last = data_ask
            logdata_bid_last = logdata_bid
            logdata_ask_last = logdata_ask

            ctr_data += 1

        elif ((ctr_data > 0) and (time <= end_time - (ctr_data * time_incr))) :
               
            rv_bid += np.power((logdata_bid - logdata_bid_last),2)
            rv_ask += np.power((logdata_ask  - logdata_ask_last),2)
            rcv_bidask += np.multiply((logdata_ask - logdata_ask_last), (logdata_bid - logdata_bid_last))
            
            data_bid_last = data_bid
            data_ask_last = data_ask
            logtv_bid_last = logdata_bid
            logdata_ask_last = logdata_ask
            timelast = time            

            ctr_data += 1

            
        if time < start_time:
            break
                
    rv_bid /= (end_time - start_time)
    rv_ask /= (end_time - start_time)
    rcv_bidask /= (end_time - start_time)

    return rcv_bidask, rv_bid, rv_ask


