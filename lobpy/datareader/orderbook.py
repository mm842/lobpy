"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller

This module provides the (not on stand-alone working) class OBReader which defines a kind of interface for classes used to read in limit order book data and return the relevant parts as numpy arrays. Note that no specific format of the data is required and thus, OBReader does not provide full functionability. This should be provided by a suitable subclass for each specific data format. (see lobster.py for a reader for LOBSTERDATA files).

"""



######
# Imports
######
import csv
import warnings 

import math
import numpy as np
#import ... as ...

######
# Global Constants
######

AGLOBAL = 0

ORDERBOOK_FILE_ID = "orderbook"
MESSAGE_FILE_ID = "message"
AV_ORDERBOOK_FILE_ID = "av-orderbook"
TVOLPROC_FILE_ID = "ordervolume"
DEPTH_FILE_ID = "marketdepth"
PRICE_FILE_ID = "best_prices"
######
# Begin Exceptions
######

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class DataFileError(Error):
    """Exception raised for errors concerning the data file.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

    def __str__(self):
        return ': '.join((self.expression, self.message))

class DataRequestError(Error):
    """Exception raised for errors concerning the data file.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self,  message):
        self.message = message

class FunctionCallError(Error):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
    def __str__(self):
        return ': '.join((self.expression, self.message))

######
# Begin Functions
######


class OBReader():
    """
    The OBReader offers support to extract features from limit order book data.
    ---------- 
    params:
        lobfilename:    name of the file where the order book data are stored
        data:    dict object which contains the data.
        num_levels:    number of price levels to consider. If None, it is expected that this will be clear from data
    """


    def __init__(self):
        self.lobfilename = "lobfilename"
        self.data = dict()
        self.num_levels = None
        
    def set_obfilename(filename):
        self.lobfilename = filename
        return True

    def create_filestr(self, identifier_str, num_levels=None):
        """ Create a file string which allows unified file names """
        return "_".join(("OBReader", identifier_str, num_levels))
    
    def get_data_from_file(self, filename=""):
        """ Loads the order book data from a csv file with format bidprofile, askprofile """
        if filename=="":
            filename=self.lobfilename
        try:
            with open(filename+".csv", newline='') as file:
                filedata = csv.reader(file,delimiter=',')
                bid_profile = next(filedata)
                ask_profile = next(filedata)
        except FileNotFoundError:
            warnings.warn("File not found %s, returning None, None"%filename)
            return None, None
        
        return np.fromiter(bid_profile, np.float), np.fromiter(ask_profile, np.float)
    
    
    def add_data(self, key_str, data, overwrite=True):
        """
        Adds data to the data dict.
        ---------
        args:
            key_str:    Key to store the data
            data:       data to be stored
            overwrite:  If False and data are stored already in key, then a warning will be raised and False will be returned
        """
        
        if not overwrite and (key_str in self.data):
            warnings.warn("LOBSTERReader has already data stored with key {} and overwriting disabled.".format(key_str),RuntimeWarning)
            return False
        self.data[key_str] = data
        return True

    def remove_data(self, key_str):
        """ Removes data stored with key key_str from data dict """
        if self.data.pop(key_str) is None:
            warnings.warn("No key {} found. Nothing done.".format(key_str), RuntimeWarning)
            return False
        return True

    def clear(self):
        self.data.clear()
        return True

    def average_profile_tt(self, num_levels_output="" , write_outputfile=False):
        pass;

    def average_profile_from_file(self, num_levels_calc_str="", write_outputfile=False):
        pass;

    def _load_ordervolume(
            self,
            num_observations,
            num_levels_calc,
            profile2vol_fct=np.sum
    ):
        """ This provides an interface for a function which should implement effectively how the data are loaded from the specific file structure. 
        
        Note: Due to possibly large amount of data we iterate through the file instead of reading the whole file into an array.
        """
        raise FunctionCallError("OBReader._load_ordervolume", "This function is constructed as an interface and not thought for use in runtime. Please make sure the OBReader object has implemented _load_overvolume before usage.")
        dt = .1
        time_stamps = np.zeros(num_observations)
        volume_bid = np.zeros(num_observations)
        volume_ask = np.zeros(num_observations)
        
        return dt, time_stamps, volume_bid, volume_ask


    def _load_ordervolume_levelx(
            self,            
            level_x,
            level
    ):
        ''' Extracts the volume of orders in the first num_level buckets at a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The volume process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file. profile2vol_fct allows to specify how the volume should be summarized from the profile. Typical choices are np.sum or np.mean.

        Note: Due to possibly large amount of data we iterate through the file instead of reading the whole file into an array. 
        '''
        raise FunctionCallError("OBReader._load_ordervolume", "This function is constructed as an interface and not thought for use in runtime. Please make sure the OBReader object has implemented _load_overvolume before usage.")
        dt = .1
        time_stamps = np.zeros(num_observations)
        volume_bid = np.zeros(num_observations)
        volume_ask = np.zeros(num_observations)
        
        return dt, time_stamps, volume_bid, volume_ask



    def _load_ordervolume_full(
            self,
            num_levels_calc,
            profile2vol_fct=np.sum,
            ret_np=True
    ):
        ''' Extracts the volume of orders in the first num_level buckets from the interval [time_start_calc, time_end_calc].  profile2vol_fct allows to specify how the volume should be summarized from the profile. Typical choices are np.sum or np.mean. If ret_np==False then the output format are lists, else numpy arrays

        Note: Due to possibly large amount of data we iterate through the file instead of reading the whole file into an array. 
        '''
        pass;
        
    def load_ordervolume_levelx(
            self,
            num_observations,
            level_x,
            write_output=False):
        ''' Extracts the volume of orders in the level_xth buckets on bid and ask time at a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The volume process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file. If num_obersvations is None, then the volume will be extracted on the time grid given in the data'''

        print("Start extraction of order volume")

        if level_x > self.num_levels:
            raise DataRequestError("Level {} is not included in the data file.".format(str(level_x)))

        dt, time_stamps, volume_bid, volume_ask = self._load_ordervolume_levelx(num_observations, level_x)

        self.add_data("time_discr_ov", dt)

        self.add_data("time_stamps_ov", time_stamps)
        self.add_data("ordervolume_level-" + str(level_x) + "--bid", volume_bid)
        self.add_data("ordervolume_level-" + str(level_x) + "--ask", volume_ask)

        print("Extraction of order volume process finished.")

        if write_output:
            print("Writing output.")
            outfilename =  self.create_filestr(TVOLPROC_FILE_ID , "level-"+str(level_x))
            outfilename = ".".join((outfilename,'csv'))
            with open(outfilename, 'w') as outfile:
                wr = csv.writer(outfile)
                wr.writerow(['Time in sec', 'Volume Bid', 'Volume Ask'])
                wr.writerows(zip(time_stamps,
                                 volume_bid,
                                 volume_ask))

        return dt, time_stamps, volume_bid, volume_ask

    def load_ordervolume(
            self,
            num_observations,
            num_levels_calc_str="",
            write_output=False):
        ''' Extracts the volume of orders in the first num_level buckets at a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The volume process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file. If num_obersvations is None, then the volume will be extracted on the time grid given in the data'''

        print("Start extraction of order volume")

        num_levels_calc = self.num_levels
        if not(num_levels_calc_str == ""):
            num_levels_calc = int(num_levels_calc_str)        

        if num_observations is None:
            time_stamps, volume_bid, volume_ask = self._load_ordervolume_full(num_levels_calc, profile2vol_fct=np.sum, ret_np=False)
            dt = 0
        else:
            dt, time_stamps, volume_bid, volume_ask = self._load_ordervolume(num_observations, num_levels_calc, profile2vol_fct=np.sum)
            self.add_data("time_discr_ov", dt)

        self.add_data("time_stamps_ov", time_stamps)
        self.add_data("ordervolume" + str(num_levels_calc) + "--bid", volume_bid)
        self.add_data("ordervolume" + str(num_levels_calc) + "--ask", volume_ask)

        print("Extraction of order volume process finished.")

        if write_output:
            print("Writing output.")
            outfilename =  self.create_filestr(TVOLPROC_FILE_ID , str(num_levels_calc))
            outfilename = ".".join((outfilename,'csv'))
            with open(outfilename, 'w') as outfile:
                wr = csv.writer(outfile)
                wr.writerow(['Time in sec', 'Volume Bid', 'Volume Ask'])
                wr.writerows(zip(time_stamps,
                                 volume_bid,
                                 volume_ask))

        return dt, time_stamps, volume_bid, volume_ask

    
    def load_volume_process(
            self,
            num_observations,
            num_levels_calc_str="",
            write_output=False):
        ''' Extracts the volume of orders in the first num_level buckets at a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The volume process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file.  '''
        warnings.warn("Function load_volume_process will be removed soon. Please use load_ordervolume instead", FutureWarning)
        return self.load_ordervolume(num_observations, num_levels_calc_str, write_output)


    def load_marketdepth(
            self,
            num_observations,
            num_levels_calc_str="",            
            write_output=False
    ):
        ''' Extracts the market depths at the top of the book by averaging the order volume in the first num_level buckets. The data are extracted on a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The volume process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file. '''
        num_levels_calc = self.num_levels
        if not(num_levels_calc_str == ""):
            num_levels_calc = int(num_levels_calc_str)

        # The function _load_ordervolume will load the data according to the specific format
        if num_observations is None:            
            time_stamps, volume_bid, volume_ask = self._load_ordervolume_full(num_levels_calc, profile2vol_fct=np.mean, ret_np=False)
            dt = 0
        else:           
            dt, time_stamps, depth_bid, depth_ask = self._load_ordervolume(num_observations, num_levels_calc, profile2vol_fct=np.mean)
            self.add_data("time_discr_depth", dt)
            
        self.add_data("time_stamps_depth", time_stamps)
        self.add_data("market_depth" + str(num_levels_calc) + "--bid", depth_bid)
        self.add_data("market_depth" + str(num_levels_calc) + "--ask", depth_ask)

        print("Extraction of market depth processes finished.")

        if write_output:
            print("Writing output.")
            outfilename =  self.create_filestr(DEPTH_FILE_ID , str(num_levels_calc))
            outfilename = ".".join((outfilename,'csv'))
            with open(outfilename, 'w') as outfile:
                wr = csv.writer(outfile)
                wr.writerow(['Time in sec', 'Depth Bid', 'Depth Ask'])
                wr.writerows(zip(time_stamps,
                                 depth_bid,
                                 depth_ask))

        return dt, time_stamps, depth_bid, depth_ask

    def _load_prices(self, num_observations):
        ''' private method to implement how the price data are loaded from the files. WARNING: Not implemented in OBReader!'''
        pass;
    
    def load_prices(
            self,
            num_observations,
            write_output=False
    ):
        ''' Extracts the price on bid and ask side as well as the mid price at a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file. 
        '''
        dt, time_stamps, prices_bid, prices_ask = self._load_prices(num_observations)
        
        self.add_data("time_stamps_prices", time_stamps)
        self.add_data("prices" + "--bid", prices_bid)
        self.add_data("prices" + "--ask", prices_ask)

        print("Extraction of best price processes finished.")

        if write_output:
            print("Writing output.")
            outfilename =  self.create_filestr(PRICE_FILE_ID)
            outfilename = ".".join((outfilename,'csv'))
            with open(outfilename, 'w') as outfile:
                wr = csv.writer(outfile)
                wr.writerow(['Time in sec', 'Price Mid', 'Price Bid', 'Price Ask'])
                wr.writerows(zip(time_stamps,
                                 np.add(prices_bid, prices_ask) / float(2),
                                 prices_bid,
                                 prices_ask))
        return dt, time_stamps, prices_bid, prices_ask


    def load_profile_snapshot(
            self,
            time_stamp,
            num_levels_calc=None
    ):
        ''' Returns a two numpy arrays with snapshots of the bid- and ask-side of the order book at a given time stamp
        Output:
        bid_prices, bid_volume, ask_prices, ask_volume
        '''             
        return np.empty(1), np.empty(1), np.empty(1), np.empty(1)
    
# END OBReader




def get_data_from_file(filename):
    ''' Loads the order book data from a csv file with format bidprofile; askprofile '''
    warnings.warn("The method get_data_from_file is supposed to be removed in future versions. Use the OBReader function with the same name instead.", FutureWarning)
    try:
        with open(filename+".csv", newline='') as file:
            filedata = csv.reader(file,delimiter=',')
            bid_profile = next(filedata)
            ask_profile = next(filedata)
    except FileNotFoundError:
        print("File not found %s"%filename)
        return None, None
    
    return np.fromiter(bid_profile, np.float), np.fromiter(ask_profile, np.float)
