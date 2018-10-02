"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller

price.py 

Provides the class PriceModel and possibly additional methods for price modeling. 
"""


import math
import numpy as np

from lobpy.models.loblinear import LOBLinearTwoFactor
from lobpy.models.loblinear import OrderVolumeMeanRev
from lobpy.models.loblineartools import LOBProfile
from lobpy.models.loblineartools import LOBFactorProcess
from lobpy.models.calibration import OrderVolumeCalibrator
from lobpy.models.calibration import ParameterHistory
PRICEMODEL_STR = "PriceModel"

def _simulate_price_bachelier(dt, num_tpoints, p0, mu, sigma):
    """ Simulate path of the bachelier model
    p(t) = p0 + mu * t + sigma * Bt
    """
    out = np.empty(num_tpoints+1)
    out[0] = p0
    wt =  np.cumsum(math.sqrt(dt) * np.random.normal(size=num_tpoints))
    out[1:] = p0 + mu * np.linspace(dt, dt*num_tpoints, num_tpoints) + sigma * wt
    return out




class PriceModel(OrderVolumeMeanRev):
    """
    This class provides the price models which are induced by the linear SPDELOB models in loblinear.py
    ---------
    fields:
        modelid:        identifier string
        profile_bid:    bid profile, LOBProfile object
        profile_ask:    ask profile, LOBProfile object
        dynamics_bid:   bid dynamics of depth, LOBFactorProcess object
        dynamics_ask:   ask dynamics of depth, LOBFactorProcess object
        rho:            correlation between bid and ask side
    """

    def __init__(
            self,
            modelid="PriceModelID",
            num_levels=2,
            p0=0.,
            tick_size=1/float(100)
    ):
        """
        Optional parameter for initialization:
            modelid="PriceModel",    name of the price model
            num_levels=2,            number of levels considered for depths modeling
            p0=0.                    initial price level
        """
        
        super().__init__(modelid=("_".join((PRICEMODEL_STR, modelid))), num_levels=num_levels)
        self.p0 = p0
        self.tick_size = tick_size

    def set_p0(self, p0):
        """ Set mid price p0 """
        self.p0 = p0
        return

    def set_depth(self, depth_bidask):
        """ Set the initial depths on bid and ask side 
            arg: depth_bidask = (depth_bid, depth_ask)
            """
        self.set_z0(depth_bidask)
        return
    
    def get_depth(self):
        """ Returns the initial depth """
        return self.get_z0()

    def get_vol(self): 
        """ Returns the mid price volatility. """
        sigma = self.get_sigma()
        price_vol = (math.sqrt((math.pow(sigma[0],2) + math.pow(sigma[1], 2) - 2 * sigma[0] * sigma[1] * self.rho)) / float(2))
        return (price_vol * self.tick_size)
    
    ###############################
    # Predictions tools
    ###############################
    
    def simulate_depth(self, dt, num_tpoints, return_bmincr=False):
        """ Simulate a trijectory of the depth for bid and ask side using Euler scheme. This function calls simulate_dynamics from OrderVolumeMeanRev.
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
        return self.simulate_dynamics(dt, num_tpoints, return_bmincr)
    
    
    def simulate(self, dt, num_tpoints, out=None, ret_depths=False):
        """ Simulate a trijectory of the price, with grid size dt and using num_tpoints future time points. Optionally one can give a numpy array out of length num_tpoints + 1 in which the path will be stored. If ret_depths is True then also the simulated depths processes for bid and ask side are returned."""
        #np.random.seed()

        dt = float(dt)
        vol = self.get_vol()
        mu = self.get_mu()
        nu = self.get_nu()
        sigma = self.get_sigma()
        rho2 = math.sqrt(1 - math.pow(self.rho,2))
        
        if not ret_depths and (mu[0] == 0) and (mu[1] == 0):
            return _simulate_price_bachelier(dt, num_tpoints, self.p0, (nu[1]- nu[0]) / float(2), vol)
        
        depth_bid, depth_ask, dw1t, dw2t = self.simulate_depth(dt, num_tpoints, return_bmincr=True)
        if out is None:
            out = np.empty(num_tpoints+1)

        out[0] = self.p0
        for k in range(num_tpoints):
            out[k+1] = out[k] + dt / float(2)  * (nu[0] * mu[0] / depth_bid[k] - nu[1] * mu[1] / depth_ask[k] - (nu[0] - nu[1]) )  + (sigma[0] - self.rho * sigma[1]) * dw1t[k] / float(2) - rho2 * sigma[1] * dw2t[k] / float(2)
            
        if ret_depths:
            return out, depth_bid, depth_ask
        
        return out

    def prob_upmove(self, t1, threshold=0.):     
        ''' Returns the probability that the price is larger than the threshold at time point t1 '''
        
        d0b, d0a = self.get_depth()
        vol = self.get_vol()
        mu = self.get_mu()
        nu = self.get_nu()
        sqt1 = math.sqrt(t1)
        prob = norm.cdf(sqt1/ (float(2.) * vol) * ( nu[0] * (mu[0] - d0b) / d0b - nu[1] * (mu[1]-d0a) / d0a) - threshold / (vol * sqt1))

        return prob
    
        




