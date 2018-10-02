"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller

 lobpy - model - loblineartools
 Helper for linear SPDE models
 Version: v0.1
"""


######
# Imports
######
import math
import warnings

import numpy as np
import scipy.optimize as sopt
import scipy




class LOBProfile:
    """ Describes a profile of the linear two factor order book model """
    
    def __init__(self, bidask = 'ask', gamma = 0.):
        """ Creates and LOBProfile object """
        self.pmax = 1000.
        self.gamma = gamma # Ratio of convection and diffusion part in the model
        self.bidask = bidask # Bid-ask flag
        return
    
    def settobid(self):
        """ Sets bidask flag to bid """
        self.bidask = 'bid'
        return
    
    def settoask(self):
        """ Sets bidask flag to ask """    
        self.bidask = 'ask'
        return
    
    def is_bid(self):
        """ Returns true if self is a bid profile """
        if self.bidask == 'bid':
            return True
        return False
    
    def is_ask(self):
        """ Returns true if self is an ask profile """
        if self.bidask == 'ask':
            return True
        return False
    

    # Profile functions
    
    # Private
    def _profilemass(self, gamma):
        """ Returns the integral from 0 to L of sin(pi/L x) exp(-gamma x). """
        L = self.pmax
        return (np.exp(-gamma * L) + 1) / (np.power(gamma,2) * L / np.pi  + np.pi / L ) 

    def _normalizevolume(self, gamma):
        """ Returns the integral from 0 to L of sin(pi/L x) exp(-gamma x). """
        L = self.pmax
        return   (np.power(gamma,2) * L / np.pi  + np.pi / L ) / (float(np.exp(-gamma * L) + 1))


    def _fit_profile_tv_leastsq_full(self, profile_data):
        """ Fit average order book profile to the model shape by using least square optimization after fitting the volume """
        
        len_data = len(profile_data)
        L = self.pmax
        
        # check if the data size is too large for the fit. In this case, reduce the size of the array which is taken into account.
        if len_data > .5 * L:
            len_data = int(len_data / 2)
            print("Warning: Parameter L was too small. Size of data array has been reduced to %i"%len_data)
            profile_data = profile_data[:len_data:] 
            
        def res_profile(par):
            """ residual profile to fit to """
            gamma = par[0]
            z0 = par[1]
            x_arr =  np.linspace(1, len_data, len_data) - .5
            res = profile_data - z0 * np.sin(np.pi / L * x_arr) * np.exp(-gamma * x_arr) 
            return res

        pargmax = np.argmax(profile_data) +.5   # position (in number of ticks) of the maximum 
        gamma = 1. / (float(pargmax)) # starting value for the optimization
        z0 = np.sum(profile_data)
        par = sopt.leastsq(res_profile, np.array((gamma, z0)))
        gamma = par[0][0]
        z0 = par[0][1]
        del par
        return gamma, z0    


    # Public

    
    def get_profilemass(self):
        return self._profilemass(self.gamma)
    

    
    def fit_profile_tv_leastsq(self, profile_data):
        """ Fit average order book profile to the model shape by using least square optimization after fitting the volume """
        
        len_data = len(profile_data)
        L = self.pmax
        
        profile_data_normalized = profile_data / float(np.sum(profile_data))   # Normalize data to fit
        
        # check if the data size is too large for the fit. In this case, reduce the size of the array which is taken into account.
        if len_data > .5 * L:
            len_data = int(len_data / 2)
            print("Warning: Parameter L was too small. Size of data array has been reduced to %i"%len_data)
            profile_data_normalized = profile_data[:len_data:] / float(np.sum(profile_data[:len_data:]))
            
        def res_profile(par):
            """ residual profile to fit to """
            gamma = par[0]
            x_arr =  np.linspace(1, len_data, len_data) - .5
            res = profile_data_normalized - np.sin(np.pi / L * x_arr) * np.exp(-gamma * x_arr) / self._profilemass(gamma)
            return res

        pargmax = np.argmax(profile_data) +.5   # position (in number of ticks) of the maximum 
        gamma = 1. / (float(pargmax)) # starting value for the optimization
        par = sopt.leastsq(res_profile, np.array([gamma]))
        self.gamma = par[0][0]
        del par
        return self.gamma

    
    def fit_profile_tv_max_exact(self, profile_data):
        """ Fit average order book profile to the model shape by fitting to the posistion of the maximum """
        
        L = self.pmax
        pargmax = np.argmax(profile_data) + .5   # position (in number of ticks) of the maximum 
        self.gamma = math.pi / (L * math.tan(math.pi * float(pargmax) / L))
        return self.gamma

    def fit_profile_tv_argmax(self, profile_data):
        """ Fit average order book profile to the model shape by fitting to the posistion of the maximum """
        
        L = self.pmax
        pargmax = np.argmax(profile_data) + .5   # position (in number of ticks) of the maximum 
        self.gamma = 1. / float(pargmax)
        return self.gamma


    
    def fit_profile_tvrmax1(self, profile_data):
        """ Fit average order book profile to the model shape by fitting R_{1,infty}. """
        
        L = self.pmax
        tvol = np.sum(profile_data)     # total volume in the data
        pargmax =np.argmax(profile_data) + .5
        
        def r1max(gamma):
            if gamma < 0:
                return float('NaN')
            if gamma == 0:
                return math.pi * 0.5
            else:
                return (math.sqrt(math.pow(L * gamma,2) + math.pow(math.pi,2)) * math.exp(- gamma * L  * math.atan(math.pi / (L * gamma)) / math.pi) / (1. + math.exp(- gamma * L / math.pi)) / L)

        r1max_res = lambda gamma : math.pow(np.max(profile_data) / tvol - r1max(gamma), 2)        
        gamma = sopt.minimize_scalar(r1max_res, bounds = (0,.5*L),method='bounded')
        self.gamma = gamma.x
        return self.gamma


class LOBProfileScaled:
    """ Describes a profile of the linear two factor order book model """
    
    def __init__(self, bidask = 'ask', gamma = 0., scaling=1.0):
        """ Creates and LOBProfile object """
        self.pmax = 1000.
        self.gamma = gamma # Ratio of convection and diffusion part in the model
        self.powerscaling = scaling
        self.bidask = bidask # Bid-ask flag
        return


    def _profile_model_scaled(self, xlevels, gamma, z0, alph):
        ''' model profile with scaling '''
        return z0 * np.sin(np.pi / self.pmax * np.power(xlevels, alph)) * np.exp(-gamma * np.power(xlevels,alph)) 

    def _profile_model_scaled_jac(self, xlevels, gamma, z0, alph):
        ''' jacobi matrix of _profile_model_scaled '''
        xlevels_rescaled = np.power(xlevels, alph)
        ddgamma = - xlevels_rescaled * z0 * np.sin(np.pi / self.pmax * xlevels_rescaled) * np.exp(-gamma * xlevels_rescaled)
        ddz0 = np.sin(np.pi / self.pmax * xlevels_rescaled) * np.exp(-gamma * xlevels_rescaled)
        ddalph = -gamma  * z0 * np.sin(np.pi / self.pmax * xlevels_rescaled) * np.exp(-gamma * xlevels_rescaled)
        ddalph += np.pi / self.pmax * z0 * np.cos(np.pi / self.pmax * np.power(xlevels, alph)) * np.exp(-gamma * xlevels_rescaled)
        ddalph = np.multiply(ddalph,np.multiply(np.log(xlevels), xlevels_rescaled))
        return np.transpose([ddgamma, ddz0, ddalph])
    
    
    def fit_profile_leastsq_scaling(self, profile_data):
        """ Fit average order book profile to the model shape by using least square optimization after fitting the volume """
        
        len_data = len(profile_data)
        L = self.pmax
        
        # check if the data size is too large for the fit. In this case, reduce the size of the array which is taken into account.
        if len_data > .5 * L:
            len_data = int(len_data / 2)
            print("Warning: Parameter L was too small. Size of data array has been reduced to %i"%len_data)
            profile_data = profile_data[:len_data:] 
            
        # def res_profile(par):
        #     """ residual profile to fit to """
        #     gamma = par[0]
        #     z0 = par[1]
        #     alph = par[2]
        #     x_arr =  np.linspace(1, len_data, len_data) - .5
        #     res = profile_data - z0 * np.sin(np.pi / L * np.power(x_arr, alph)) * np.exp(-gamma * np.power(x_arr,alph)) 
        #     return res
        x_arr =  np.linspace(1, len_data, len_data) - .5
        pargmax = np.argmax(profile_data) +.5   # position (in number of ticks) of the maximum 
        gamma = 1. / (float(pargmax)) # starting value for the optimization
        z0 = np.sum(profile_data)
        try:
            par = sopt.curve_fit(
                self._profile_model_scaled,
                x_arr, profile_data,
                p0=np.array((gamma, z0, 1.)),
                method='trf',
                bounds=((0,0,0),
                        (np.inf,np.inf,np.inf)
                ),
                jac=self._profile_model_scaled_jac
            )
        except TypeError as e:
            # jac supported from scipy version 0.17 onwards
            print(e)            
            warnings.warn("Optimization failed due to an exception. Note that Jacobi supported from scipy version 0.18 onwards. The current scipy version is: {}".format(scipy.__version__))
            try:
                par = sopt.curve_fit(
                self._profile_model_scaled,
                    x_arr, profile_data,
                    p0=np.array((gamma, z0, 1.)),
                    method='trf',
                    bounds=((0,0,0),
                            (np.inf,np.inf,np.inf)
                    )
                )
            except TypeError as e2:
                print(e)            
                warnings.warn("Optimization failed due to an exception. Running optimization without boundary constraints on the parameters now. Please check the results carefully. Note that bounds are supported from scipy version 0.17 onwards. The current scipy version is: {}".format(scipy.__version__))             
                # bounds supported from scipy version 0.18 onwards
                par = sopt.curve_fit(
                    self._profile_model_scaled,
                    x_arr, profile_data,
                    p0=np.array((gamma, z0, 1.)),
                    method='trf'
                )

                
        gamma = par[0][0]
        z0 = par[0][1]
        alph = par[0][2]
        self.gamma = gamma
        self.powerscaling = alph
        del par
        return gamma, z0, alph


    def fit_profile_leastsq_scaling_1(self, profile_data):
        """ Fit average order book profile to the model shape by using least square optimization after fitting the volume """
        
        len_data = len(profile_data)
        L = self.pmax
        
        # check if the data size is too large for the fit. In this case, reduce the size of the array which is taken into account.
        if len_data > .5 * L:
            len_data = int(len_data / 2)
            print("Warning: Parameter L was too small. Size of data array has been reduced to %i"%len_data)
            profile_data = profile_data[:len_data:] 
            
        def res_profile(par):
            """ residual profile to fit to """
            gamma = par[0]
            z0 = par[1]
            alph = par[2]
            x_arr =  np.linspace(1, len_data, len_data) - .5
            res = profile_data - - z0 * np.sin(np.pi / L * np.power(x_arr, alph)) * np.exp(-gamma * np.power(x_arr,alph)) 
            return res

        pargmax = np.argmax(profile_data) +.5   # position (in number of ticks) of the maximum 
        gamma = 1. / (float(pargmax)) # starting value for the optimization
        z0 = np.sum(profile_data)
        par = sopt.leastsq(res_profile, np.array((gamma, z0, 1.)))
        par = sopt.curve_fit(res_profile, np.array((gamma, z0, 1.)))
        gamma = par[0][0]
        z0 = par[0][1]
        alph = par[0][2]
        self.gamma = gamma
        self.powerscaling = alph
        del par
        return gamma, z0, alph
    
    
        
### End LOBProfile

class LinearDiffusion:
    """ Class of stochastic processes of the form
        d Z_t = nu( mu - Z_t) d t + sigma Z_t d W_t
    for a real Brownian motion W and nu, mu, sigma in RR.
    """

    def __init__(self, z0 = 1., mu = 1., nu = 0., sigma = 0.):
        self.z0 = z0
        self.mu = mu
        self.nu = nu
        self.sigma = sigma

    def set_z0(self, z0):
        self.z0 = z0
        return True

    def get_z0(self):
        return self.z0

    def set_mu(self, mu):
        self.mu = mu
        return True

    def get_mu(self):
        return self.mu

    def set_nu(self, nu):
        self.nu = nu
        return True

    def get_nu(self):
        return self.nu

    def set_sigma(self, sigma):
        self.sigma = sigma
        return True

    def get_sigma(self):
        return self.sigma


    def set_params(self, params):
        ''' Set model parameter (mu, nu, sigma) to input '''
        self.mu = params[0]
        self.nu = params[1]
        self.sigma = params[2]
        return True    

    def get_params(self):
        ''' Returns the parameters of the dynamics: (mu, nu, sigma) '''
        return (self.get_mu, self.get_nu, self.sigma)
        
    
    def is_gbm(self):
        """ Returns true if the process is a geometric Brownian motion with drift """
        if (self.z0 < 0):
            warnings.warn("Negative initial value, z0 = {}".format(self.z0))
            return False
        if (self.mu == 0):
            return True
        return False

    def is_recgamma(self):
        """ Returns true if the process is a reciprocal gamma diffusion """
        if (self.z0 < 0):
            warnings.warn("Negative initial value, z0 = {}".format(self.z0))
            return False
        if (self.mu >0) and (self.nu >0):
            return True
        return False    

    def simulate(self, time_discr, num_tpoints, out=None):
        """ Returns a sample path as a numpy array computed by Euler method.
        ----------
        args:
        time_discr     difference between to time points
        num_tpoints    number of future time points for simulation
        out            (optional) array in which the simulation is stored

        output:        numpy array of length num_tpoints+1 with simulated path
        """
        if out is None:
            out = np.empty(num_tpoints+1)
        dt = float(time_discr)
        sqdt = math.sqrt(dt)
        #np.random.seed()
        dwt = sqdt * np.random.normal(size=num_tpoints)
        print(str(self.z0))
        out[0] = self.z0
        for k in range(num_tpoints):
            out[k+1] = out[k] + (self.mu - self.nu * out[k]) * dt + self.sigma * out[k] * dwt[k]
        
        return out
        
    
    
    
class LOBFactorProcess(LinearDiffusion):
    """ Describes a factor process of the linear two factor order book model 
    """
    
    def __init__(self, bidask = 'ask', z0 = 1., mu = 0., nu = 0., sigma = 0.):
        super().__init__(z0=z0, mu=mu, nu=nu, sigma=sigma)
        self.bidask = bidask
        return


    def settobid(self):
        """ Sets bidask flag to bid """
        self.bidask = 'bid'
        return
    
    def settoask(self):
        """ Sets bidask flag to ask """    
        self.bidask = 'ask'
        return
    
    def is_bid(self):
        """ Returns true if self is a bid profile """
        if self.bidask == 'bid':
            return True
        return False
    
    def is_ask(self):
        """ Returns true if self is an ask profile """
        if self.bidask == 'ask':
            return True
        return False
  


    # def calibrate_to_LOBster(self,ticker="SPY",date="2016-06-10", start_time_data=34200000, end_time_data=57600000, start_time_calc=34200000, end_time_calc=57600000, num_levels_data=50, num_levels_calc=2, num_time_points=3600):
    #     print('Starting calibration. Loading files.')

    #     filename_body
    #     filenameBody= ticker +'_' + date+'_' + str(int(start_time_data))+'_' + str(int(end_time_data))
    #     filenameOrderBook = fileNameBody +'_orderbook_'+ str(int(depth_data)) + '.csv'
    #     filenameMessages =  fileNameBody +'_messages_'+ str(int(depth_data)) + '.csv'

    #     t, Yb, Ya = lob.load_volume(ticker="SPY",date="2016-06-10", start_time_data=34200000, end_time_data=57600000, start_time_calc=34200000, end_time_calc=57600000, num_levels_data=50, num_levels_calc=2, num_time_points=3600, writeOutput = False)

    #     dt = t[1] - t[0]    
        
    #     self.calibrateToDepth(Yb, Ya, dt)
        
    #     print("Calibration finished")
        
    #     return 1


### End FactorProcess



# def simulate(self, time_discr, num_tpoints):
#     """ Returns a sample path as a numpy array computed by Euler method """
#     out = np.zeros(num_tpoints)
#     dt = float(time_discr)
#     sqdt = math.sqrt(dt)
#     #np.random.seed()
#     dwt = sqdt * np.random.normal(size=num_tpoints-1)
#     out[0] = self.z0
#     for k in range(num_tpoints-1):
#         out[k+1] = out[k] + (self.mu - self.nu * out[k]) * dt + self.sigma * out[k] * dwt[k]
        
#         return out
