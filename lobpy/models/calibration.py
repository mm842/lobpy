"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller

calibration.py

Contains objects and functions for calibration of the LOB models.

To calibrate dynamics to a given time series of data on bid and ask side, data_bid and data_ask, with time stamps time_stamps which is uniform with time increment time_incr, e. g. use

>>> cal = OrderVolumeCalibrator()

>>> cal.calibrate(time_stamps, time_stamps[1] - time_stamps[0], data_bid, data_ask)


"""


######
# Imports
######
import copy
from collections import defaultdict
import csv
import json
import math
import warnings

import numpy as np
import scipy.optimize as sopt
import pandas as pd



from lobpy.models.loblineartools import LOBProfile
from lobpy.models.loblineartools import LOBFactorProcess
from lobpy.models.loblinear import LOBLinear
from lobpy.models.loblinear import LOBLinearTwoFactor
from lobpy.models.loblinear import OrderVolumeMeanRev

from lobpy.models import estimators as est

CORRELATION_PAR_STR = "Correlation"


class ParameterHistory():
    """ Log of parameters for linear SPDE market models
    -------
    time_stamps: list of time stamps
    params_bid: defaultdict with lists of parameters of bid side
    params_ask: defaultdict with lists of parameters of ask side
    params_correlation: list with correlation values
    """
    
    def __init__(self):        
        self.time_stamps = []
        self.params_bid = defaultdict(list)
        self.params_ask = defaultdict(list)
        self.params_correlation = []

    def __str__(self):
        return str(self.to_list())
    
    def _check_len(self):
        """ Checks if each list in the history object has the same length """
        num_t = len(self.time_stamps)
        for key_bid, list_bid, key_ask, list_ask in zip(self.params_bid.items(), self.params_ask.items()):
            if (num_t != len(list_bid)):
                warnings.warn("Size of time stamps is not consistent with lengths of {}".format(key_ask))
                return False
            if (num_t != len(list_ask)):
                warnings.warn("Size of time stamps is not consistent with lengths of {}".format(key_ask))
                return False            

        return True

    def to_list(self, paramlist=None):
        """
        Returns parameters in form of a list of lists of format.
        ----------
        args:
        paramlist=None    list of form ['outparamkey1', outparamkey2'] in which output is written

        Output:
        [['Time', timepoint1, timepoint2,...], [param1_bid, param1_bid[timepoint1],...],...]
        if paramlist is not None, then 
        [['Time', timepoint1, timepoint2,...], [ outparamkey1', outparamvalue1[timepoint1],...],..., ['rho', ...]
        """
        outvals = [["Time"]]
        outvals[0] = outvals[0] + self.time_stamps
        if paramlist is None:
            for ctr, (key, item) in enumerate(self.params_bid.items()):
                outvals.append([key+"_bid"] + item)
                outvals.append([key+"_ask"] + self.params_ask[key])
        else:
            for key in paramlist:
                outvals.append([key+"_bid"] + self.params_bid[key])
                outvals.append([key+"_ask"] + self.params_ask[key])

        outvals.append([CORRELATION_PAR_STR] + self.params_correlation)
        return outvals

    def to_dict(self, paramlist=None):
        """
        Returns parameters in form of a dict of lists. 
        ----------
        args:
        paramlist=None    list of form ['outparamkey1', outparamkey2'] with keys which should be considered. Time and Corellation are always included.
   
        """
        out = dict()
        out['Time'] = self.time_stamps
        if paramlist is None:
            for ctr, (key, item) in enumerate(self.params_bid.items()):
                out[key+"_bid"] = item
                out[key+"_ask"] = self.params_ask[key]
        else:
            for key in paramlist:
                out[key+"_bid"] = self.params_bid[key]
                out[key+"_ask"] = self.params_ask[key]
        out[CORRELATION_PAR_STR] = self.params_correlation
        return out

    def get_modelpardict_t(self, time_index, modelid="model_from_calhist"):
        """
        Returns the model parameter as a dict given a time index time_index for the model history
        """
        out = {key: (self.params_bid[key][time_index], self.params_ask[key][time_index]) for key in self.params_bid.keys()}
        out['rho'] = self.params_correlation[time_index]
        out['modelid'] = modelid + "_tind_" + str(time_index)
        return out
    
    def append(self, time_stamp, two_factor_model):
        """ adds parameter from two factor model with given time stamp to the history """
            
        self.time_stamps.append(float(time_stamp))
        param_dict = two_factor_model.get_modelpar_dict()        
        for key, val in param_dict.items():
            
            if key in ('modelid', 'pmax'):
                continue            
            if (key == 'rho'):
                self.params_correlation.append(val)
                continue
            
            self.params_bid[key].append(val[0])
            self.params_ask[key].append(val[1])
            
        return True    
            
    def savef(self, filename, csv=True, paramlist=None):
        """ Saves the model history to a file in format:
        a) csv format if csv==True using pandas or 
        b) json format, else 
        The columns are [time, bid parameters, ask parameters, correlation]. 
        paramlist can be optionally given to fix the order and selection of parameters. See also to_list for.
        """
        outvals = self.to_list(paramlist=paramlist)
        
        if csv:
            data_frame = pd.DataFrame(outvals)
            data_frame = data_frame.transpose()
            data_frame.to_csv(filename+'.csv', index=False, header=False)
        else:
            with open(filename, "w") as outfile:
                try: json.dumps(outvals, outfile)
                except TypeError:
                    print("Unable to serialize parameter history")
                    return False
        return True

    def loadf_json(self, filename):
        """ Loads the model history to a file from json format. Warning: This method overwrites the history object.
        """
        if (len(self.time) > 0):
            warnings.warn("Loading history from file {} overwrites history object".format(filename))
            
        with open(filename, "r") as infile:
            invals = json.load(infile)
            self.time_stamps = invals[0]["Time"]
            self.params_bid = invals[1]
            self.params_ask = invals[2]
            self.params_correlation = invals[3][CORRELATION_PAR_STR]
        return True
####
def read_calhistory_csv(filename):
    """ 
    Loads the model history to a file from csv format and returns a CalibrationHistory object.
    """
    out = CalibrationHistory()
    
    invals = pd.read_csv(filename)
    out.time_stamps = invals["Time"].tolist()
    out.params_correlation = invals[CORRELATION_PAR_STR].tolist()
    for key_baflag in invals.columns:
        if key_baflag in ("Time", CORRELATION_PAR_STR):
        #these values are read in already
            continue
        elif "_" not in key_baflag:
            warnings.warn("Unkown column found in file: {}. Values will be ignored.".format(key_baflag), RuntimeWarning)
            continue
        try:
            key, baflag = key_baflag.split("_")
        except ValueError:
            warnings.warn("Unkown column found in file: {}. Expected: key_bid or key_ask. Values will be ignored.".format(key_baflag), RuntimeWarning)
            continue
        if baflag == "bid":
            out.params_bid[key] = invals[key_baflag].tolist()
        elif baflag == "ask":
            out.params_ask[key] = invals[key_baflag].tolist()                
        else:
            warnings.warn("Unkown column found in file: {1}. {2} should be bid or ask. Values will be ignored.".format(key_baflag, baflag), RuntimeWarning)
            
    return out

class CalibrationHistory(ParameterHistory):
    pass;

class PredictionHistory(ParameterHistory):
    pass;


class ModelCalibrator():
    """ This class is designed to calibrate linear models and store its history
    """
    def __init__(self):
        self.calibratorid = "ModelCalibrator"        
        self.model = LOBLinear()
        self.history = CalibrationHistory()

    
    def set_id(self, calibratorid):
        self.calibratorid = calibratorid
        return

    def get_id(self):
        return self.calibratorid
        
    def savef_history(self, csv=True):
        """ Saves the calibration history. """
        self.history.savef(self.get_id(), csv)
        return True

    def loadf_history_json(self, filename):
        """ Loads calibration history from file """
        self.history.loadf_json(filename)
        return True

    def calibrate(
            self,
            time_stamp,
            data_bid,
            data_ask
    ):
        """ Calibration to data history for bid and ask side 
        ---------
        args:
            time_stamp:   time point in float
            data_bid:      data on bid side at the time points
            data_ask:      data on ask side at the time points
        output:
            modelpar:      returns the calibrated parameters
        """
        pass;

    
class OrderVolumeCalibrator(ModelCalibrator):
    """ This class calibrates a model and stores its history
    ---------
    fields:
    calibratorid:        identifier string
    model:               current model to be calibrated
    estimator_dynamics:  function for estimation of (nu, mu, sigma) 
    estimator_corr: function for estimation of correlation (rho), if None, rho is set to 0.
    estimator_dyn_corr   If this estimator is set then it is assumed to return all parameters, incl. correlation

    """
    
    def __init__(
            self,
            calibratorid="OrderVolumeCalibrator",
            model=None,            
            estimator_dynamics=est.estimate_recgamma_diff,
            estimator_corr=None,
            estimator_dyn_corr=None
    ):
        self.calibratorid = calibratorid
        self.model = model        
        if model is None:
            self.model = OrderVolumeMeanRev()
        self.history = CalibrationHistory()
        self.estimator_dynamics = estimator_dynamics
        self.estimator_corr = estimator_corr
        self.estimator_full=estimator_dyn_corr


    def _calibrate_full(
            self,
            time_stamp,
            time_incr,
            data_bid,
            data_ask
    ):
        """ Calibration to data history for bid and ask side estimating all parameter with one estimator function
        ---------
        args:
            time_stamp:   time point in float
            time_incr:     time increments in the array
            data_bid:      volume on bid side at the time points
            data_ask:      volume on ask side at the time points
            n_start:       starting index for the calibration
            n_today:         first index not to include for the calibration (None, if calibration to full array)

        ---------
        output:
            (params_bid, params_ask, rho):      returns the calibrated parameters, each in format (mu, nu, sigma) 
        """
        params_bid, params_ask, rho = self.estimator_full(data_bid, data_ask, time_incr)
        self.model.dynamics_bid.set_params(params_bid)
        self.model.dynamics_ask.set_params(params_ask)        
        self.model.set_rho(rho)
        self.history.append(time_stamp, self.model)      
        return params_bid, params_ask, rho
        
        
    def calibrate(
            self,
            time_stamp,
            time_incr,
            data_bid,
            data_ask
    ):
        """ Calibration to data history for bid and ask side 
        ---------
        args:
            time_stamp:   time point in float
            time_incr:     time increments in the array
            data_bid:      volume on bid side at the time points
            data_ask:      volume on ask side at the time points
            n_start:       starting index for the calibration
            n_today:         first index not to include for the calibration (None, if calibration to full array)

        ---------
        output:
            (params_bid, params_ask, rho):      returns the calibrated parameters, each in format (mu, nu, sigma) 
        if estimator_full is set, then the output will be the output of that function
        """

        if not (self.estimator_full is None):
            return self._calibrate_full(
                time_stamp,
                time_incr,               
                data_bid,
                data_ask
            )
        
        
        self.model.set_z0((data_bid[-1], data_ask[-1]))
        
        params_bid = self.estimator_dynamics(data_bid, time_incr)
        self.model.dynamics_bid.set_params(params_bid)
        
        params_ask = self.estimator_dynamics(data_ask, time_incr)
        self.model.dynamics_ask.set_params(params_ask)

        if self.estimator_corr is None:
            self.model.set_rho(0.)
        else:
            self.model.set_rho(self.estimator_corr(data_bid, data_ask))
            
        self.history.append(time_stamp, self.model)      
        
        return (params_bid, params_ask, self.model.get_rho())
        

    def calibrate_running_frame(
            self,
            time_start,
            time_discr,
            data_bid,
            data_ask,
            num_timepoints_calib,
            num_timepoints_recal=1,
    ):
        """
        This function creates a price model induced by the mean reverting order book model and calibrates this model on a defined running time fram
        ----------------
        args:
        time_start:          float time point at which data starts (calibration will start num_timepoints_calib later)
        time_discr:                float time between 2 time points
        data_bid:                  data bid side (uniform time grid, starting at time_start)
        data_ask:                  data ask side (uniform time grid, starting at time_start)
        num_timepoints_calib:      number of time points to be used for each calibration
        num_timepoints_recal=1:    number of time points after which recalibration starts
        
        """
        
        # Convert to correct data type
        time_start = float(time_start)
        time_discr = float(time_discr)
        num_timepoints_calib = int(num_timepoints_calib)
        num_timepoints_recal = int(num_timepoints_recal)       
        
        print("Start calibration in time frame: {}".format(self.calibratorid))
        
        for ctr_now in range(num_timepoints_calib-1, len(data_bid), num_timepoints_recal):
            
            # Calibrate 
            # time_now = time_start + (ctr_start + num_timepoints_calib) * time_discr
            self.calibrate(
                time_start + ctr_now * time_discr,
                time_discr,
                data_bid[ctr_now - num_timepoints_calib+1:ctr_now+1:],
                data_ask[ctr_now - num_timepoints_calib+1:ctr_now+1:]                    
            )
        return


    def savef_history(self, csv=True):
        """ Saves the calibration history. """
        self.history.savef(self.get_id(), csv, paramlist=["z0", "mu", "nu", "sigma"])
        return True


class LOBProfileCalibrator(ModelCalibrator):
    """ This class calibrates a model and stores its history
    ---------
    fields:
        calibratorid:        identifier string
        fitting_method:      valid choices are "LSQ", "TVLSQ", "TV-ArgMax", "TV-Rmax1"
    """


    def __init__(
            self,
            calibratorid="LOBProfileCalibrator",
            fitting_method="LSQ",
            model=None
    ):
        self.calibratorid = calibratorid
        self.fitting_method = fitting_method
        if model is None:
            self.model = LOBLinearTwoFactor()
        else:
            self.model = model
        self.history = CalibrationHistory()


    def calibrate(self, time_stamp, data_bid, data_ask):
        """ calibrates to data by fitting the method specified in .fitting_method
        ----------
        args:
            data_bid
            data_ask:    bid and ask profiles to fit data to (array format)
        """
        if self.fitting_method == "LSQ":
            self.model.init_leastsq(data_bid, data_ask)
        if self.fitting_method == "LSQScaling":
            self.model.init_leastsq(data_bid, data_ask, scaling=True)             
        elif self.fitting_method == "TVLSQ":
            self.model.init_tv_leastsq(data_bid, data_ask)            
        elif self.fitting_method == "TV-ArgMax":
            self.model.init_tv_argmax(data_bid,data_ask)
        elif self.fitting_method == "TV-Rmax1":            
            self.model.init_tv_rmax1(data_bid, data_ask)

        self.history.append(time_stamp, self.model)
        return self.model.get_z0(), self.model.get_gamma()



    

def fit_profile_to_data(bidprofile, askprofile):
    ''' Creates and 4 LOBLinearTwoFactor models and fits them to the bid and ask profiles '''
    warnings.warn("Method fit_profile_to_data might be removed in future.", FutureWarning)
    
    model_lsq = LOBLinearTwoFactor()
    model_lsqf = LOBLinearTwoFactor()
    model_argmax = LOBLinearTwoFactor()
    model_r_max1 = LOBLinearTwoFactor()

    model_lsq.init_tv_leastsq(bidprofile, askprofile)
    model_lsqf.init_leastsq(bidprofile, askprofile)
    model_argmax.init_tv_argmax(bidprofile, askprofile)
    model_r_max1.init_tv_rmax1(bidprofile, askprofile)
    
    return model_lsq, model_lsqf, model_argmax, model_r_max1





