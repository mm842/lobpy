"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller

This module provides the helper functions and the class LOBSTERReader, a subclass of OBReader to read in limit order book data in lobster format. 
"""


######
# Imports
######
import csv
import math
import warnings 

import numpy as np

from lobpy.datareader.orderbook import *




# LOBSTER specific file name functions

def _split_lobster_filename(filename): 
    """ splits the LOBSTER-type filename into Ticker, Date, Time Start, Time End, File Type, Number of Levels """
    filename2,_ = filename.split(".")
    ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels = filename2.split("_") 
    return ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels

def split_lobster_filename(filename):
    """ splits the LOBSTER-type filename into Ticker, Date, Time Start, Time End, File Type, Number of Levels """    
    return _split_lobster_filename(filename)

def _split_lobster_filename_core(filename):
    """ splits the LOBSTER-type filename into Ticker, Date, Time Start, Time End, File Type, Number of Levels """
    filename2, _ = filename.split(".")
    ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels = filename2.split("_") 
    return ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels

    
def _create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels):
    return "_".join((ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels))


def create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels):
    return _create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels)

def _get_time_stamp_before(time_stamps, time_stamp):
    ''' Returns the value and index of the last time point in time_stamps before or equal time_stamp '''
    time = time_stamps[0]
    index = int(0)
    if time == time_stamp:
        # time_stamp found at index 0    
        return time, index
    if time > time_stamp:
        raise LookupError("Time stamp data start at {} which is after time_stamps: {}".format(time, time_stamp))
    for ctr, time_now in enumerate(time_stamps[1:]):
        if time_now > time_stamp:
            return time, ctr
        time = time_now
    
    return time, ctr+1


class LOBSTERReader(OBReader):
    """
    OBReader object specified for using LOBSTER files
    ---------- 
    params:
            ticker_str,
            date_str,
            time_start_str,
            time_end_str,
            num_levels_str,
            time_start_calc_str,
            time_end_calc_str


    Example usage:
    to create an object
    >>> lobreader = LOBSTERReader("SYMBOL", "2012-06-21", "34200000", "57600000", "10")
    read market depth on uniform time grid with num_observation number of observations
    >>> dt, time_stamps, depth_bid, depth_ask = lobreader.load_marketdepth(num_observations)
    read price process on that time grid specified above
    >>> dt2, time_stamps2, price_mid, price_bid, price_ask = lobreader.load_marketdepth(None)

    """
    
    def __init__(
            self,
            ticker_str,
            date_str,
            time_start_str,
            time_end_str,
            num_levels_str,
            time_start_calc_str="",
            time_end_calc_str="",
            num_levels_calc_str=""
    ):
        self.ticker_str = ticker_str
        self.date_str = date_str        
        self.lobfilename = _create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str, ORDERBOOK_FILE_ID, num_levels_str)
        self.msgfilename = _create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str, MESSAGE_FILE_ID, num_levels_str)
        self.time_start = int(time_start_str)
        self.time_end = int(time_end_str)
        self.num_levels = int(num_levels_str)
        self.time_start_calc = int(time_start_str)
        self.time_end_calc = int(time_end_str)
        self.num_levels_calc = int(num_levels_str)
        if not (num_levels_calc_str == ""):
            self.num_levels_calc = int(num_levels_calc_str)
        self.data = dict()        
        if not (time_start_calc_str == ""):
            self.time_start_calc = int(time_start_calc_str)
        if not (time_end_calc_str == ""):
            self.time_end_calc = int(time_end_calc_str)

 

    def set_timecalc(self, time_start_calc_str, time_end_calc_str):
        self.time_start_calc = int(time_start_calc_str)
        self.time_end_calc = int(time_end_calc_str)
        return True
        
    def create_filestr(self, identifier_str, num_levels=None):
        """ Creates lobster type file string """
        if num_levels is None:
            num_levels = self.num_levels
        return _create_lobster_filename(self.ticker_str, self.date_str, str(self.time_start_calc), str(self.time_end_calc), identifier_str, str(num_levels))

    
    def average_profile_tt(self, num_levels_calc_str="" , write_outputfile = False):
        """ Computes the average order book profile, averaged over trading time, from the csv sourcefile. To avoid numerical errors by summing up large numbers, the Kahan Summation algorithm is used for mean computation 
        ----------
        args:
            num_levels_calc:    number of levels which should be considered for the output
            write_output:         if True, then the average order book profile is stored as a csv file
        ----------
        output:
            (mean_bid, mean_ask)  in format of numpy arrays
        """
        
        print("Starting computation of average order book profile in file %s."%self.lobfilename)
        
        num_levels_calc = self.num_levels

        if not(num_levels_calc_str == ""):
            num_levels_calc = int(num_levels_calc_str)
            
        if self.num_levels < num_levels_calc:
            raise DataRequestError("Number of levels in data ({0}) is smaller than number of levels requested for calculation ({1}).".format(self.num_levels, num_levels_calc))
        tempval1 = 0.0
        tempval2 = 0.0
        comp = np.zeros(num_levels_calc * 2)     # compensator for lost low-order bits
        mean = np.zeros(num_levels_calc * 2)    # running mean

        with open(self.lobfilename+".csv", newline='') as csvfile:
            lobdata = csv.reader(csvfile, delimiter=',')
            num_lines = sum(1 for row in lobdata)
            print("Loaded successfully. Number of lines: " + str(num_lines))    
            csvfile.seek(0)         # reset iterator to beginning of the file
            print("Start calculation.")
            for row in lobdata:     # data are read as list of strings
                currorders = np.fromiter(row[1:(4*num_levels_calc + 1):2], np.float)    # parse to integer                
                for ctr, currorder in enumerate(currorders):
                #print(lobstate)
                    tempval1 = currorder / num_lines - comp[ctr]
                    tempval2 = mean[ctr] + tempval1
                    comp[ctr] = (tempval2 - mean[ctr]) - tempval1
                    mean[ctr] = tempval2
                    
            print("Calculation finished.")

            # Add data to self.data
            self.add_data("--".join(("ttime-"+AV_ORDERBOOK_FILE_ID, "bid")), mean[1::2])
            self.add_data("--".join(("ttime-"+AV_ORDERBOOK_FILE_ID, "ask")), mean[0::2])            
    
            if not write_outputfile:
                return mean[1::2], mean[0::2] # LOBster format: bid data at odd * 2,  LOBster format: ask data at even * 2

            print("Write output file.")    
            outfilename = self.create_filestr("-".join(("ttime",AV_ORDERBOOK_FILE_ID)) , str(num_levels_calc))
            outfilename = ".".join((outfilename,'csv'))
            with open(outfilename, 'w') as outfile:
                wr = csv.writer(outfile)
                wr.writerow(mean[1::2]) # LOBster format: bid data at odd * 2
                wr.writerow(mean[0::2]) # LOBster format: ask data at even * 2
                
            print("Average order book saved as %s."%outfilename)
            return mean[1::2], mean[0::2] 
                    



    def average_profile(
            self,
            num_levels_calc_str="",
            write_outputfile = False                
    ):
        """ Returns the average oder book profile from the csv sourcefile, averaged in real time. To avoid numerical errors by summing up large numbers, the Kahan Summation algorithm is used for mean computation  """
        
        
        if num_levels_calc_str == "":
            num_levels_calc = self.num_levels_calc
        else:
            num_levels_calc = int(num_levels_calc_str)
            
        if int(self.num_levels) < num_levels_calc:
            raise DataRequestError("Number of levels in data ({0}) is smaller than number of levels requested for calculation ({1}).".format(self.num_level, num_levels_calc))
                
        time_start = float(self.time_start_calc / 1000.)
        time_end = float(self.time_end_calc / 1000.)
        mean = np.zeros(num_levels_calc * 2)    # running mean
        tempval1 = 0.0
        tempval2 = 0.0
        linectr = 0
        comp = np.zeros(num_levels_calc * 2)     # compensator for lost low-order bits
        flag = 0
    
        with open(".".join((self.lobfilename, 'csv')), newline='') as orderbookfile, open(".".join((self.msgfilename, 'csv')), newline='') as messagefile:
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')        
                
            rowMES = next(messagedata)      # data are read as list of strings
            rowLOB = next(lobdata)        
            nexttime = float(rowMES[0])   # t(0)
            if time_end < nexttime:
                # In this case there are no entries in the file for the selected time interval. Array of 0s is returned
                warnings.warn("The first entry in the data files is after the end of the selected time period. Arrays of 0s will be returned as mean.")
                return mean[1::2], mean[0::2]            
            currprofile =  np.fromiter(rowLOB[1:(4*num_levels_calc + 1):2], np.float)    # parse to integer, extract bucket volumes only at t(0)
            if time_start <= nexttime:
                flag = 1
                    
            for rowLOB, rowMES in zip(lobdata,messagedata):     # data are read as list of string, iterator now starts at second entry (since first has been exhausted above)
                currtime = nexttime    #(t(i))
                nexttime = float(rowMES[0])      #(t(i+1))             
                if flag == 0:
                    if time_start <= nexttime:
                        # Start calculation
                        flag = 1
                        currtime = time_start
                        
                        for ctr, currbucket in enumerate(currprofile):
                            tempval1 = (nexttime - currtime) / float(time_end - time_start) * currbucket - comp[ctr]
                            tempval2 = mean[ctr] + tempval1
                            comp[ctr] = (tempval2 - mean[ctr]) - tempval1
                            mean[ctr] = tempval2
                else:
                    if time_end < nexttime:
                        # Finish calculation
                        nexttime = time_end
                        
                    for ctr, currbucket in enumerate(currprofile):
                        #print(currprofile)
                        tempval1 = (nexttime - currtime) / float(time_end - time_start) * currbucket - comp[ctr]
                        tempval2 = mean[ctr] + tempval1
                        comp[ctr] = (tempval2 - mean[ctr]) - tempval1
                        mean[ctr] = tempval2

                    if time_end == nexttime:
                        # Finish calculation                    
                        break
                
                ## Update order book to time t(i+1)
                currprofile =  np.fromiter(rowLOB[1:(4*num_levels_calc + 1):2],np.float)    # parse to integer, extract bucket volumes only                               
            else:  # executed only when not quitted by break, i.e. time_end >= time at end of file in this case we extrapolate
                warnings.warn("Extrapolated order book data since time_end exceed time at end of the file by %f seconds."%(time_end - nexttime))
                currtime = nexttime
                nexttime = time_end
                for ctr, currbucket in enumerate(currprofile):
                    #print(lobstate)
                    tempval1 = (nexttime - currtime) / (time_end - time_start) * currbucket - comp[ctr]
                    tempval2 = mean[ctr] + tempval1
                    comp[ctr] = (tempval2 - mean[ctr]) - tempval1
                    mean[ctr] = tempval2                

        print("Calculation finished.")

        # Add data to self.data
        self.add_data("--".join((AV_ORDERBOOK_FILE_ID, "bid")), mean[1::2])
        self.add_data("--".join((AV_ORDERBOOK_FILE_ID, "ask")), mean[0::2])            
        
        if not write_outputfile:
            return mean[1::2], mean[0::2] # LOBster format: bid data at odd * 2,  LOBster format: ask data at even * 2

        print("Write output file.")    
        outfilename =  self.create_filestr(AV_ORDERBOOK_FILE_ID , str(num_levels_calc))
        outfilename = ".".join((outfilename,'csv'))
        with open(outfilename, 'w') as outfile:
            wr = csv.writer(outfile)
            wr.writerow(mean[1::2]) # LOBster format: bid data at odd * 2
            wr.writerow(mean[0::2]) # LOBster format: ask data at even * 2
            
        print("Average order book saved as %s."%outfilename)
        return mean[1::2], mean[0::2] 
        


        

    def _load_ordervolume(
            self,
            num_observations,
            num_levels_calc,
            profile2vol_fct=np.sum
    ):
        ''' Extracts the volume of orders in the first num_level buckets at a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The volume process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file. profile2vol_fct allows to specify how the volume should be summarized from the profile. Typical choices are np.sum or np.mean.

        Note: Due to possibly large amount of data we iterate through the file instead of reading the whole file into an array. 
        '''


        time_start_calc = float(self.time_start_calc) / 1000.
        time_end_calc = float(self.time_end_calc) / 1000.
        file_ended_line = int(num_observations)
        ctr_time = 0
        ctr_line = 0
        ctr_obs = 0   # counter for the outer of the        
        time_stamps, dt = np.linspace(time_start_calc, time_end_calc, num_observations, retstep = True)
        volume_bid = np.zeros(num_observations)
        volume_ask = np.zeros(num_observations)

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')        
            # get first row
            # data are read as list of strings
            rowMES = next(messagedata)
            rowLOB = next(lobdata)
            # parse to float, extract bucket volumes only
            currprofile = np.fromiter(rowLOB[1:(4*num_levels_calc + 1):2], np.float)
            time_file = float(rowMES[0])

            for ctr_obs, time_stamp in enumerate(time_stamps):
                if (time_stamp < time_file):
                    # no update of volume in the file. Keep processes constant
                    if (ctr_obs > 0):
                        volume_bid[ctr_obs] = volume_bid[ctr_obs-1] 
                        volume_ask[ctr_obs] = volume_ask[ctr_obs-1]
                    else:
                        # so far no data available, raise warning and set processes to 0.
                        warnings.warn("Data do not contain beginning of the monitoring period. Values set to 0.", RuntimeWarning)
                        volume_bid[ctr_obs] = 0.
                        volume_ask[ctr_obs] = 0.
                    continue

                while(time_stamp >= time_file):
                    # extract order volume from profile
                    volume_bid[ctr_obs] = profile2vol_fct(currprofile[1::2])
                    volume_ask[ctr_obs] = profile2vol_fct(currprofile[0::2])

                    # read next line
                    try:
                        rowMES = next(messagedata)      # data are read as list of strings
                        rowLOB = next(lobdata)                
                    except StopIteration:
                        if (file_ended_line == num_observations):
                            file_ended_line = ctr_obs
                        break
                    # update currprofile and time_file
                    currprofile = np.fromiter(rowLOB[1:(4*num_levels_calc + 1):2], np.float)    # parse to integer, extract bucket volumes only
                    time_file = float(rowMES[0])                        

        if (file_ended_line < num_observations):
            warnings.warn("End of file reached. Number of values constantly extrapolated: %i"%(num_observations - file_ended_line), RuntimeWarning)


        return dt, time_stamps, volume_bid, volume_ask


    def _load_ordervolume_levelx(
            self,            
            num_observations,
            level
    ):
        ''' Extracts the volume of orders in the first num_level buckets at a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The volume process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file. profile2vol_fct allows to specify how the volume should be summarized from the profile. Typical choices are np.sum or np.mean.

        Note: Due to possibly large amount of data we iterate through the file instead of reading the whole file into an array. 
        '''


        time_start_calc = float(self.time_start_calc) / 1000.
        time_end_calc = float(self.time_end_calc) / 1000.
        file_ended_line = int(num_observations)
        ctr_time = 0
        ctr_line = 0
        ctr_obs = 0   # counter for the outer of the        
        time_stamps, dt = np.linspace(time_start_calc, time_end_calc, num_observations, retstep = True)
        volume_bid = np.zeros(num_observations)
        volume_ask = np.zeros(num_observations)

        # Ask level x is at position (x-1)*4 + 1, bid level x is at position (x-1)*4 + 3
        x_bid = (int(level) - 1) * 4 + 3
        x_ask = (int(level) - 1) * 4 + 1
        

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')        
            # get first row
            # data are read as list of strings
            rowMES = next(messagedata)
            rowLOB = next(lobdata)
            # parse to float, extract bucket volumes only

            #currprofile = np.fromiter(rowLOB[1:(4*num_levels_calc + 1):2], np.float)
            currbid = float(rowLOB[x_bid])
            currask = float(rowLOB[x_ask])
            time_file = float(rowMES[0])

            for ctr_obs, time_stamp in enumerate(time_stamps):
                if (time_stamp < time_file):
                    # no update of volume in the file. Keep processes constant
                    if (ctr_obs > 0):
                        volume_bid[ctr_obs] = volume_bid[ctr_obs-1] 
                        volume_ask[ctr_obs] = volume_ask[ctr_obs-1]
                    else:
                        # so far no data available, raise warning and set processes to 0.
                        warnings.warn("Data do not contain beginning of the monitoring period. Values set to 0.", RuntimeWarning)
                        volume_bid[ctr_obs] = 0.
                        volume_ask[ctr_obs] = 0.
                    continue

                while(time_stamp >= time_file):
                    # extract order volume from profile
                    volume_bid[ctr_obs] = currbid
                    volume_ask[ctr_obs] = currask

                    # read next line
                    try:
                        rowMES = next(messagedata)      # data are read as list of strings
                        rowLOB = next(lobdata)                
                    except StopIteration:
                        if (file_ended_line == num_observations):
                            file_ended_line = ctr_obs
                        break
                    # update currprofile and time_file
                    #currprofile = np.fromiter(rowLOB[1:(4*num_levels_calc + 1):2], np.float)    # parse to integer, extract bucket volumes only
                    currbid = float(rowLOB[x_bid])
                    currask = float(rowLOB[x_ask])
                    time_file = float(rowMES[0])                        

        if (file_ended_line < num_observations):
            warnings.warn("End of file reached. Number of values constantly extrapolated: %i"%(num_observations - file_ended_line), RuntimeWarning)


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
        time_start_calc = float(self.time_start_calc) / 1000.
        time_end_calc = float(self.time_end_calc) / 1000.

        time_stamps = []
        volume_bid = [] 
        volume_ask = [] 
        index_start = -1
        index_end = -1
        
        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')        
            # get first row
            # data are read as list of strings

            for ctrRow, (rowLOB, rowMES) in enumerate(zip(lobdata, messagedata)):
                time_now = float(rowMES[0])
                if (index_start == -1) and (time_now >= time_start_calc):
                    index_start = ctrRow
                if (index_end == -1) and (time_now > time_end_calc):
                    index_end = ctrRow 
                    break
                    
                time_stamps.append(time_now)                    
                currprofile = np.fromiter(rowLOB[1:(4*num_levels_calc + 1):2], np.float)    # parse to integer, extract bucket volumes only
                volume_bid.append(profile2vol_fct(currprofile[1::2]))
                volume_ask.append(profile2vol_fct(currprofile[0::2]))
        
        if index_end == -1:
            #file end reached
            index_end = len(time_stamps)
            
        if ret_np:
            return np.array(time_stamps[index_start:index_end]), np.array(volume_bid[index_start:index_end]), np.array(volume_ask[index_start:index_end])
        return time_stamps[index_start:index_end], volume_bid[index_start:index_end], volume_ask[index_start:index_end]



    def _load_prices(
            self,
            num_observations
    ):
        ''' private method to implement how the price data are loaded from the files '''
        time_start_calc = float(self.time_start_calc) / 1000.
        time_end_calc = float(self.time_end_calc) / 1000.
        file_ended_line = int(num_observations)
        ctr_time = 0
        ctr_line = 0
        ctr_obs = 0   # counter for the outer of the        
        time_stamps, dt = np.linspace(time_start_calc, time_end_calc, num_observations, retstep = True)
        prices_bid = np.empty(num_observations)
        prices_ask = np.empty(num_observations)

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')        
            # get first row
            # data are read as list of strings
            rowMES = next(messagedata)
            rowLOB = next(lobdata)
            time_file = float(rowMES[0])

            for ctr_obs, time_stamp in enumerate(time_stamps):
                if (time_stamp < time_file):
                    # no update of prices in the file. Keep processes constant
                    if (ctr_obs > 0):
                        prices_bid[ctr_obs] = prices_bid[ctr_obs-1] 
                        prices_ask[ctr_obs] = prices_ask[ctr_obs-1]
                    else:
                        # so far no data available, raise warning and set processes to 0.
                        warnings.warn("Data do not contain beginning of the monitoring period. Values set to 0.", RuntimeWarning)
                        prices_bid[ctr_obs] = 0.
                        prices_ask[ctr_obs] = 0.
                    continue

                while(time_stamp >= time_file):
                    # LOBster stores best ask and bid price in resp. 1st and 3rd column, price in unit USD*10000
                    prices_bid[ctr_obs] = float(rowLOB[2]) / float(10000) 
                    prices_ask[ctr_obs] = float(rowLOB[0]) / float(10000)
                

                    # read next line
                    try:
                        rowMES = next(messagedata)      # data are read as list of strings
                        rowLOB = next(lobdata)                
                    except StopIteration:
                        if (file_ended_line == num_observations):
                            file_ended_line = ctr_obs
                        break
                    # update time_file
                    time_file = float(rowMES[0])                        

        if (file_ended_line < num_observations-1):
            warnings.warn("End of file reached. Number of values constantly extrapolated: %i"%(num_observations - file_ended_line), RuntimeWarning)
            while ctr_obs < (num_observations-1):
                prices_bid[ctr_obs+1] = prices_bid[ctr_obs]
                prices_ask[ctr_obs+1] = prices_ask[ctr_obs]


        return dt, time_stamps, prices_bid, prices_ask

    def _load_profile_snapshot_lobster(
            self,
            time_stamp,
            num_levels_calc=None
    ):
        ''' Returns a two numpy arrays with snapshots of the bid- and ask-side of the order book at a given time stamp
        Output:
        bid_prices, bid_volume, ask_prices, ask_volume
        '''
        #convert time from msec to sec
        time_stamp = float(time_stamp) / 1000.

        if num_levels_calc is None:
            num_levels_calc = self.num_levels_calc

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')        
            # get first row
            # data are read as list of strings
            rowMES = next(messagedata)
            rowLOB = next(lobdata)
            # parse to float, extract bucket volumes only
            time_file = float(rowMES[0])
            if time_file > time_stamp:
                raise LookupError("Time data in the file start at {} which is after time_stamps: {}".format(time_file, time_stamp))
            if time_file == time_stamp:
                # file format is [ask level, ask volume, bid level, bid volume, ask level, ....]
                #conversion of price levels to USD
                bid_prices = np.fromiter(rowLOB[2:(4*num_levels_calc):4], np.float) / float(10000)
                bid_volume = np.fromiter(rowLOB[3:(4*num_levels_calc):4], np.float)
                #conversion of price levels to USD
                ask_prices = np.fromiter(rowLOB[0:(4*num_levels_calc):4], np.float) / float(10000)
                ask_volume = np.fromiter(rowLOB[1:(4*num_levels_calc):4], np.float)               
                
            for rowMES in messagedata:
                time_file = float(rowMES[0])
                if time_file > time_stamp:
                    # file format is [ask level, ask volume, bid level, bid volume, ask level, ....]
                    #conversion of price levels to USD
                    bid_prices = np.fromiter(rowLOB[2:(4*num_levels_calc):4], np.float) / float(10000)
                    bid_volume = np.fromiter(rowLOB[3:(4*num_levels_calc):4], np.float)
                    #conversion of price levels to USD
                    ask_prices = np.fromiter(rowLOB[0:(4*num_levels_calc):4], np.float) / float(10000)
                    ask_volume = np.fromiter(rowLOB[1:(4*num_levels_calc):4], np.float)
                    break
                
                rowLOB = next(lobdata)                
            else:
                # time in file did not exceed time stamp to the end. Return last entries of the file
                bid_prices = np.fromiter(rowLOB[2:(4*num_levels_calc):4], np.float) / float(10000)
                bid_volume = np.fromiter(rowLOB[3:(4*num_levels_calc):4], np.float)
                #conversion of price levels to USD
                ask_prices = np.fromiter(rowLOB[0:(4*num_levels_calc):4], np.float) / float(10000)
                ask_volume = np.fromiter(rowLOB[1:(4*num_levels_calc):4], np.float)
            return bid_prices, bid_volume, ask_prices, ask_volume
            
        
    def load_profile_snapshot(
            self,
            time_stamp,
            num_levels_calc=None
    ):
        ''' Returns a two numpy arrays with snapshots of the bid- and ask-side of the order book at a given time stamp
        Output:
        bid_prices, bid_volume, ask_prices, ask_volume
        '''             
        return self._load_profile_snapshot_lobster(time_stamp, num_levels_calc)                
    
# END LOBSTERReader


