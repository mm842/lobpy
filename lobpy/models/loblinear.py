"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller

 lobpy - model - lob_linear
 Defines classes for linear SPDE models:

 LOBLinearTwoFactor: Linear 2-factor models

 LOBLinearTwoFactorS: Linear 2-factor models with rescaling to fit the order book profile

 OrderVolumeMeanRev: Reduced model for market depths or order book volume processe only. 

 MarketMeanRev: under construction - not fully implemented yet

"""


######
# Imports
######
import csv
import json
import math
import warnings

import numpy as np
import scipy.optimize as sopt

from lobpy.models.loblineartools import LOBProfile
from lobpy.models.loblineartools import LOBProfileScaled
from lobpy.models.loblineartools import LOBFactorProcess


######
# Global Constants
######
#

##


class LOBLinear:
    """ Linear SPDE models for Limit Order Books 
    So far, this class is not implemented but allows to implement subclasses.
    """
    pass
#



######
# The following two classes will be used for a specific subclass of limit order book models which admit nice factorizations
######


class LOBLinearTwoFactor(LOBLinear):
    """Class of linear SPDE models for Limit Order Books
    in which the density of order for the bid and ask side is given as
    u(t,x) = h(x) Z^{b/a}_t
    where 
    h(x) = sin(x pi/pmax) * exp(- gamma_ask x), x>0,
    h(x) = sin(x pi/pmax) * exp(gamma_bid x), x<0,
    and
    d Z^{b/a}_t = nu_bid/ask * ( mu_bid/ask - Z_t^{b/a} ) dt + sigma_bid/ask Z_t^{b/a} d W_t^{b/a} 
    and [W^b, W^a]_t = rho * t. This corresponds to the general situation where the driving noise process is a Brownian motion and the model is started in the first eigenspace. 
 
   iHere, t describes the time point and x the price in the coordinates 
    x(p) = c_data_bid/ask * (S_t - p) / delta
    where delta is the tick size, S_t is the mid price process and p is in {delta * k : k non-negative integer}. 

    The LOBLinearTwoFactor has the following attributes
    
    modelid ...Model ID to identify the model e.g. when printing plots or saving data
    profile_bid
    profile_ask ...LOBProfile objects which describe the shape of h for bid and ask side, repsectively
    dynamics_bid
    dynamics_ask ...LOBDynamics objects which describe the dynamics governed by Z^b and Z^a
    correlation_bid_ask ...Correlation parameter rho between the Brownian motions driving the bid and ask dynamics

    If dynamics_bid.mu and dynamics_ask.mu are 0, then this class corresponds to linear homogeneous models discussed in Section 3. When mu != 0, then we are in the linear inhomgeneous / mean-reverting situation in Section 4. 

    ---------
    fields:
        modelid:        identifier string
        profile_bid:    bid profile, LOBProfile object
        profile_ask:    ask profile, LOBProfile object
        dynamics_bid:   bid dynamics, LOBFactorProcess object
        dynamics_ask:   ask dynamics, LOBFactorProcess object
        rho:            correlation between bid and ask side
    """    
    
    #
    def __init__(self, modelid="LinearModel"):
        """ parameter format: [bidparameter, askparameter] """

        self.modelid = modelid # Model ID to identify the model e.g. when printing plots or saving data
        
        self.profile_bid = LOBProfile('bid')
        self.profile_ask = LOBProfile('ask')        
        self.dynamics_bid = LOBFactorProcess('bid')
        self.dynamics_ask = LOBFactorProcess('ask')     
        self.rho = 0.           # correlation between bid and ask side


    #
    ######
    # Begin Functions
    ######

    def set_modelid(self, modelid):
        self.modelid = modelid
        return

    def set_gamma(self, gammas):
        self.profile_bid.gamma = gammas[0]
        self.profile_ask.gamma = gammas[1]
        return


    def set_z0(self, z0s):
        """ Set bid and ask initial values for z0 """
        self.dynamics_bid.z0 = z0s[0]
        self.dynamics_ask.z0 = z0s[1]
        return

    def set_nu(self, nus):
        self.dynamics_bid.nu = nus[0]
        self.dynamics_ask.nu = nus[1]
        return

    def set_mu(self, mus):
        self.dynamics_bid.mu = mus[0]
        self.dynamics_ask.mu = mus[1]
        return

    def set_sigma(self, sigmas):
        self.dynamics_bid.sigma = sigmas[0]
        self.dynamics_ask.sigma = sigmas[1]
        return

    def set_rho(self, rho):
        self.rho = rho
        return


    def set_modelpar_dict(self, pardict):
        """ sets model parameter to the parameter given in dict format 
        Touples are formatted as (bid_value, ask_value)
        """

        self.set_modelid(pardict['modelid'])
        self.profile_bid.pmax, self.profile_ask.pmax = pardict['pmax']
        self.set_gamma(pardict['gamma'])
        self.set_z0(pardict['z0']) 
        self.set_nu(pardict['nu'])
        self.set_mu(pardict['mu'])
        self.set_sigma(pardict['sigma'])
        self.set_rho(pardict['rho'])

        return 
    
    def get_modelid(self):
        return self.modelid

    def get_gamma(self):
        return self.profile_bid.gamma, self.profile_ask.gamma

    
    def get_z0(self):
        """ Returns bid and ask values for z0 """
        return self.dynamics_bid.z0, self.dynamics_ask.z0

    def get_nu(self):
        return self.dynamics_bid.nu, self.dynamics_ask.nu

    def get_mu(self):
        return self.dynamics_bid.mu, self.dynamics_ask.mu

    def get_sigma(self):
        return self.dynamics_bid.sigma, self.dynamics_ask.sigma

    def get_rho(self):
        return self.rho

    def get_modelpar_dict(self):
        """ returns the model parameters in dict format. 
        Touples are formatted as (bid_value, ask_value)
        """
        pardict = {}
        pardict['modelid'] = self.modelid
        pardict['pmax'] = self.profile_bid.pmax, self.profile_ask.pmax
        pardict['gamma'] = self.get_gamma()
        pardict['z0'] = self.get_z0()
        pardict['nu'] = self.get_nu()
        pardict['mu'] = self.get_mu()
        pardict['sigma'] = self.get_sigma()
        pardict['rho'] = self.get_rho()
        return pardict


    def simulate_dynamics(self, dt, num_tpoints, return_bmincr=False):
        """ Simulate a trijectory of the depth for bid and ask side using Euler scheme. 
        ----------
        args:
        dt     grid size
        num_tpoints     number of time points on the grid
        return_bmincr         if True, then also the sample paths of the ind. BMs used for simulation will be returned

        output:
        path_bid, path_ask     numpy arrays with the simulated paths
        or
        path_bid, path_ask, dw1t, dw2t
        """

        path_bid = np.empty(num_tpoints+1)
        path_ask = np.empty(num_tpoints+1)
        dt = float(dt)
        sqdt = math.sqrt(dt)
        dw1t = sqdt * np.random.normal(size=num_tpoints)
        dw2t = sqdt * np.random.normal(size=num_tpoints)
        rho_conv = math.sqrt(1 - math.pow(self.rho,2))
        (path_bid[0], path_ask[0]) = self.get_z0()
        for k in range(num_tpoints):
            path_bid[k+1] = path_bid[k] + self.dynamics_bid.nu * (self.dynamics_bid.mu - path_bid[k]) * dt + self.dynamics_bid.sigma * path_bid[k] * dw1t[k]
            path_ask[k+1] = path_ask[k] + self.dynamics_ask.nu * (self.dynamics_ask.mu - path_ask[k]) * dt + self.dynamics_ask.sigma * path_ask[k] * (self.rho * dw1t[k] + rho_conv * dw2t[k])

        if return_bmincr:
            return path_bid, path_ask, dw1t, dw2t

        return path_bid, path_ask
    
    def init_leastsq(self, profile_data_bid, profile_data_ask):
        """ Initialize model by fitting to data using least square fit """

        gamma_bid, z0_bid = self.profile_bid._fit_profile_tv_leastsq_full(profile_data_bid)
        gamma_ask, z0_ask = self.profile_ask._fit_profile_tv_leastsq_full(profile_data_ask)
        self.set_gamma((gamma_bid, gamma_ask))
        self.set_z0((z0_bid, z0_ask))
        return True

    
    
    def init_tv_leastsq(self, profile_data_bid, profile_data_ask):
        """ Initialize model by fitting to data as follows:
        1) initial volume is kept fixed from the data
        2) gamma is computed by least square fit
        """
        gamma_bid = self.profile_bid.fit_profile_tv_leastsq(profile_data_bid)
        gamma_ask = self.profile_ask.fit_profile_tv_leastsq(profile_data_ask)
        self.set_z0((np.sum(profile_data_bid) / self.profile_bid.get_profilemass(), np.sum(profile_data_ask)/ self.profile_ask.get_profilemass()))
        return True

    def init_tv_argmax(self, profile_data_bid, profile_data_ask):
        """ Initialize model by fitting to data as follows:
        1) initial volume is kept fixed from the data
        2) gamma is computed by position of the maximum
        """
        gamma_bid = self.profile_bid.fit_profile_tv_argmax(profile_data_bid)
        gamma_ask = self.profile_ask.fit_profile_tv_argmax(profile_data_ask)
        self.set_z0((np.sum(profile_data_bid) / self.profile_bid.get_profilemass(), np.sum(profile_data_ask)/ self.profile_ask.get_profilemass()))
        return True

    def init_tv_rmax1(self, profile_data_bid, profile_data_ask):
        """ Initialize model by fitting to data as follows:
        1) initial volume is kept fixed from the data
        2) gamma is computed by least square fit
        """
        gamma_bid = self.profile_bid.fit_profile_tvrmax1(profile_data_bid)
        gamma_ask = self.profile_ask.fit_profile_tvrmax1(profile_data_ask)
        self.set_z0((np.sum(profile_data_bid) / self.profile_bid.get_profilemass(), np.sum(profile_data_ask)/ self.profile_ask.get_profilemass()))
        return True

    
    def get_profilefct_bid(self):
        """ Returns a function with the calibrated profile of the _bid side as a function of relative price x """
        z0 = self.dynamics_bid.z0
        c2 = 1. # self.scalingconst[0]
        L = self.profile_bid.pmax
        gamma = self.profile_bid.gamma
        return (lambda x :   c2 * z0 * np.sin(np.pi / L * c2 * x) * np.exp(-gamma * c2 * np.abs(x)))

    def get_profilefct_ask(self):
        """ Returns a function with the calibrated profile of the _ask side as a function of relative price x """        
        z0 = self.dynamics_ask.z0
        c2 = 1. #self.scalingconst[1]
        L = self.profile_ask.pmax
        gamma = self.profile_ask.gamma
        return (lambda x :   c2 * z0 * np.sin(np.pi / L * c2 * x) * np.exp(-gamma * c2 * np.abs(x)))

    def get_profilefct(self):
        """ Returns the profile as a function of relative price x """
        profileFct_bid = self.get_profile_bid
        profileFct_ask = self.get_profile_ask
        return (lambda x: ((float(x<0))*profileFct_bid(x) + (float(x>0))*profileFct_ask(x)))

    
    def __str__(self):
        return str(self.get_modelpar_dict())



    def savef(self):
        """ Saves model in a file with name modelid.lobm using json """
        filename = ".".join((self.get_modelid(), "lobm"))
        with open(filename, "w") as outfile:
            try: json.dump(self.get_modelpar_dict(), outfile)
            except TypeError:
                print("Unable to serialize parameter.")
                return False
        return True
    
    def loadf(self, filename):
        """ Creates and loads a model from a file in json format with name filename.lobm """
        filenamelobm = ".".join((filename, "lobm"))
        with open(filenamelobm, "r") as infile:
            modelpar = json.load(infile)
            self.set_modelpar_dict(modelpar)
        return True
    

    ######

    ######
    # End Functions
    ######

# End LOBLinearTwoFactor


class LOBLinearTwoFactorS(LOBLinearTwoFactor):
    ''' LOBLinearTwoFactor which supports rescaling of tick level scale by x ** alpha for some alpha > 0.'''


    def __init__(self, modelid="LinearModel"):
        """ parameter format: [bidparameter, askparameter] """

        self.modelid = modelid # Model ID to identify the model e.g. when printing plots or saving data
        
        self.profile_bid = LOBProfileScaled('bid')
        self.profile_ask = LOBProfileScaled('ask')        
        self.dynamics_bid = LOBFactorProcess('bid')
        self.dynamics_ask = LOBFactorProcess('ask')     
        self.rho = 0.           # correlation between bid and ask side


    def init_leastsq(self, profile_data_bid, profile_data_ask):
        """ Initialize model by fitting to data using least square fit """

        gamma_bid, z0_bid, alph_bid = self.profile_bid.fit_profile_leastsq_scaling(profile_data_bid)
        gamma_ask, z0_ask, alph_ask = self.profile_ask.fit_profile_leastsq_scaling(profile_data_ask)
        self.set_gamma((gamma_bid, gamma_ask))
        self.set_z0((z0_bid, z0_ask))
        self.set_scaling((alph_bid, alph_ask))
        
        return True

    def set_scaling(self, alphas):
        ''' Sets the scaling of the price level grid to x ** alpha
        '''
        self.profile_bid.powerscaling = alphas[0]
        self.profile_ask.powerscaling = alphas[1]
        return    

    
    def get_scaling(self):
        return self.profile_bid.powerscaling, self.profile_ask.powerscaling
    

    def set_modelpar_dict(self, pardict):
        """ sets model parameter to the parameter given in dict format 
        Touples are formatted as (bid_value, ask_value)
        """
        super().set_modelpar_dict(pardict)
        self.set_scaling(pardict['scaling'])

        return 

    def get_modelpar_dict(self):
        """ returns the model parameters in dict format. 
        Touples are formatted as (bid_value, ask_value)
        """
        pardict = super().get_modelpar_dict()
        pardict['scaling'] = self.get_scaling()
        return pardict
    

    def get_profilefct_bid(self):
        """ Returns a function with the calibrated profile of the _bid side as a function of relative price x """
        z0 = self.dynamics_bid.z0
        c2 = 1. # self.scalingconst[0]
        L = self.profile_bid.pmax
        gamma = self.profile_bid.gamma
        alph = self.profile_ask.powerscaling
        return (lambda x :   c2 * z0 * np.sin(np.pi / L * c2 * np.power(x,alph)) * np.exp(-gamma * c2 * np.power(np.abs(x),alph)))

    def get_profilefct_ask(self):
        """ Returns a function with the calibrated profile of the _ask side as a function of relative price x """        
        z0 = self.dynamics_ask.z0
        c2 = 1. #self.scalingconst[1]
        L = self.profile_ask.pmax
        gamma = self.profile_ask.gamma
        alph = self.profile_ask.powerscaling        
        return (lambda x :   c2 * z0 * np.sin(np.pi / L * c2 * np.power(x,alph)) * np.exp(-gamma * c2 * np.power(np.abs(x),alph)))



    
    
class OrderVolumeMeanRev(LOBLinearTwoFactor):
    """ Model for the volume of orders in the first levels of the order book 

    fields:
        modelid:        identifier string
        num_levels:     integer describing the number of levels considered for computation
        dynamics_bid:   bid dynamics, LOBFactorProcess object
        dynamics_ask:   ask dynamics, LOBFactorProcess object
        rho:            correlation between bid and ask side
    
    Following fields from super class are not used here and set to None:
        profile_bid    
        profile_ask    

    """
    
    
    def __init__(self, modelid="OrderVolumeMeanRev", num_levels=1):       
        self.num_levels = num_levels
        self.modelid = "_".join((modelid, str(num_levels))) # Model ID to identify the model e.g. when printing plots or saving data
                                
        self.profile_bid = None # Pure price model does not implement profiles
        self.profile_ask = None
        
        self.dynamics_bid = LOBFactorProcess('bid')
        self.dynamics_ask = LOBFactorProcess('ask')        
        #self.c_data = [1.,1.]  # Used for linear rescaling to fit to the data

        self.rho = 0.           # correlation between bid and ask side

    
    def get_gamma(self):
        warnings.warn("Pure price model does not implement order book profiles.")
        return None

    def get_modelpar_dict(self):
        """ returns the model parameters in dict format. 
        Touples are formatted as (bid_value, ask_value)
        """
        pardict = {}
        pardict['modelid'] = self.modelid
        pardict['z0'] = self.get_z0()
        pardict['nu'] = self.get_nu()
        pardict['mu'] = self.get_mu()
        pardict['sigma'] = self.get_sigma()
        pardict['rho'] = self.get_rho()
        return pardict    


    
    def set_modelpar_dict(self, pardict):
        """ sets model parameter to the parameter given in dict format 
        Touples are formatted as (bid_value, ask_value)
        """
        self.set_modelid(pardict['modelid'])
        self.set_z0(pardict['z0']) 
        self.set_nu(pardict['nu'])
        self.set_mu(pardict['mu'])
        self.set_sigma(pardict['sigma'])
        self.set_rho(pardict['rho'])
        return

    
    ######################
    # NOT CONSIDERED FOR IMPLEMENTATION
    ######################
    
    def init_leastsq(self, profile_data_bid, profile_data_ask):
        """ Not implemented """
        warnings.warn("Function init_leastsq not implemented for subclass OrderVolumeMeanRev")
        return None

    def init_tv_leastsq(self, profile_data_bid, profile_data_ask):
        """ Not implemented """
        warnings.warn("Function init_tv_leastsq not implemented for subclass OrderVolumeMeanRev")
        return None

    def init_tv_argmax(self, profile_data_bid, profile_data_ask):
        """ Not implemented """
        warnings.warn("Function init_tv_argmax not implemented for subclass OrderVolumeMeanRev")
        return None

    def init_tv_rmax1(self, profile_data_bid, profile_data_ask):
        """ Not implemented """
        warnings.warn("Function init_tv_rmax1 not implemented for subclass OrderVolumeMeanRev")
        return None
    
    def get_profilefct_bid(self):
        """ Not implemented """
        warnings.warn("Function get_profilefct_bid not implemented for subclass OrderVolumeMeanRev")
        return None

    def get_profilefct_ask(self):
        """ Not implemented """
        warnings.warn("Function get_profilefct_ask not implemented for subclass OrderVolumeMeanRev")
        return None

    def get_profilefct(self):
        """ Not implemented """
        warnings.warn("Function get_profilefct not implemented for subclass OrderVolumeMeanRev")
        return None

    
    
    
class MarketMeanRev(LOBLinearTwoFactor):
    """ 
    TODO

    """

    def __init__(self):
        """ parameter format: [bidparameter, askparameter] """
        super().__init__()
        self.set_modelid("MarketModelMeanRev") # Model ID to identify the model e.g. when printing plots or saving data
        
        self.prop = 1.  # Sensitivity constant on the order book dynamics
        self.pmid0 = 0.  # Initial price level 





    


        
    
