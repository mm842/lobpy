"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller
handler.py


contains some functions to work with the model

"""


import math
import sys

import numpy as np
import pandas as pd

import lobpy.datareader.lobster as lob
import lobpy.datareader.lobster as lobr
import lobpy.models.loblinear as lobm
import lobpy.models.loblineartools as lobt
import lobpy.models.plots as lobp
import lobpy.models.calibration as cal
import lobpy.models.estimators as est
import lobpy.models.price as lobprice



# def calibrate_mrevdynamics_lobster_rf_f(
#         ticker_str,
#         date_str,
#         time_start_data,
#         time_end_data,
#         time_start_calc,
#         time_end_calc,
#         num_levels_data,
#         num_levels_calc,
#         timegrid_size,
#         ntimesteps_cal,
#         ntimesteps_nextcal
# ):
#     """ Calibrates mean reverting model to order book volume loaded from lobster data  """
    
#     # read files from lobster to uniform grid
#     print('Extracting total volume process on uniform grid.')    
#     dt, time_stamps, volume_bid, volume_ask = lob.load_volume_process(
#         ticker_str,
#         date_str,
#         time_start_data,
#         time_end_data,
#         time_start_calc,
#         time_end_calc,
#         num_levels_data,
#         num_levels_calc,
#         timegrid_size
#     )
#     print("Finished.")
    
#     print('Start calibration on time frame')
#     # Create calibrator object with id inherited from lobster notation and estimator for correlation based on realized covariance
#     ov_cal = cal.OrderVolumeCalibrator(
#         calibratorid=lob.create_lobster_filename(ticker_str, date_str, str(time_start_calc), str(time_end_calc), "cal_ordervolume-f", str(num_levels_calc)),
#         estimator_dynamics=est.estimate_recgamma_diff,
#         estimator_corr=est.estimate_log_corr_rv
#     )
    
#     ov_cal.calibrate_running_frame(
#         time_stamps[0],
#         dt,
#         volume_bid,
#         volume_ask,
#         ntimesteps_cal,
#         ntimesteps_nextcal
#     )
#     # save history as csv file
#     print('Calibration finished. Saving csv file.')
#     ov_cal.savef_history(csv=True)
#     print('Calibration history saved.')
    
#     # create plots


#     lobp.plot_calibration_history_volume(ov_cal.history, filename=ov_cal.calibratorid, titlestr=" ".join((ticker_str, date_str, str(num_levels_calc))))
#     print('Plots saved')


def calibrate_mrevdynamics_lobster_rf(
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        ntimepoints_grid,
        ntimesteps_cal,
        ntimesteps_nextcal,
        cal_to_average=False,
        cal_to_average_classic=False,
):
    """ Calibrates mean reverting model to order book volume loaded from lobster data  
    -----------
    args:
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        ntimepoints_grid,
        ntimesteps_cal,
        ntimesteps_nextcal,
        cal_to_average=False    calibration to total volume (if False) in the first buckets or average
        cal_to_average_classic=False    calibration to total volume (if False) in the first buckets or average - by just averaging after extraction

    """
    
    # read files from lobster to uniform grid
    lobreader = lobr.LOBSTERReader(
        ticker_str,
        date_str,
        str(time_start_data),
        str(time_end_data),
        str(num_levels_data),
        str(time_start_calc),
        str(time_end_calc)
    )
    
    print('Extracting total volume process on uniform grid.')    
    
    if cal_to_average:
        dt, time_stamps, volume_bid, volume_ask = lobreader.load_marketdepth(
            num_observations=ntimepoints_grid,
            num_levels_calc_str=str(num_levels_calc),        
            write_output=False
        )
    else:    
        dt, time_stamps, volume_bid, volume_ask = lobreader.load_ordervolume(
            num_observations=ntimepoints_grid,
            num_levels_calc_str=str(num_levels_calc),        
            write_output=False
        )
        if cal_to_average_classic:
            volume_bid = np.true_divide(volume_bid, num_levels_calc)
            volume_ask = np.true_divide(volume_ask, num_levels_calc)
            
    print("Finished.")
            
        
    print('Start calibration on time frame')
    # Create calibrator object with id inherited from lobster notation and estimator for correlation based on realized covariance
    ov_cal = cal.OrderVolumeCalibrator(
        calibratorid=lobreader.create_filestr(identifier_str="cal_ordervolume", num_levels=str(num_levels_calc)),
        estimator_dynamics=est.estimate_recgamma_diff,
        estimator_corr=est.estimate_log_corr_rv
    )
    
    ov_cal.calibrate_running_frame(
        time_stamps[0],
        dt,
        volume_bid,
        volume_ask,
        ntimesteps_cal,
        ntimesteps_nextcal
    )

    # save history as csv file
    print('Calibration finished. Saving csv file.')
    ov_cal.savef_history(csv=True)
    print('Calibration history saved.')
    
    # create plots
    lobp.plot_calibration_history_volume(ov_cal.history, filename=ov_cal.calibratorid, titlestr=" ".join((ticker_str, date_str, str(num_levels_calc))))
    print('Plots saved')
    




def calibrate_profile_lobster(
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        time_cal
):
    """ Splits the day into time intervals of specified size and fits the model profile to the average profile of the order book from lobster data. 4 different fitting methods are used. In addition, the average profile for the whole day will be fitted.
    -----------
    args:
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        time_cal            time for averageing in ms
    """

    gamma_bids_LSQ = []
    gamma_asks_LSQ = []
    
    gamma_bids_LSQF = []
    gamma_asks_LSQF = []
    
    gamma_bids_ArgMax = []
    gamma_asks_ArgMax = []
    
    gamma_bids_RMax1 = []
    gamma_asks_RMax1 = []
    

    lobreader = lobr.LOBSTERReader(
        ticker_str,
        date_str,
        str(time_start_data),
        str(time_end_data),        
        str(num_levels_data),
        str(time_start_calc),
        str(time_end_calc)
    )

    computation_interval = int(time_end_calc) - int(time_start_calc)
    num_calibration = computation_interval / int(time_cal)
    if num_calibration == 0:
        return 0
    
    time_starts = np.arange(int(time_start_calc), int(time_end_calc), int(time_cal))
    time_ends = time_starts + int(time_cal)
    if time_ends[-1] > time_end_calc:
        time_ends[-1] = time_end_calc
    for time_start, time_end in zip(time_starts, time_ends):
        print("Calibrate")        
        lobreader.set_timecalc(str(time_start), str(time_end))
        filename = lobreader.create_filestr(lobr.AV_ORDERBOOK_FILE_ID, str(num_levels_calc))
        
        av_profile_bid, av_profile_ask = lobr.get_data_from_file(filename)
        if (av_profile_bid is None) or (av_profile_ask is None):
            av_profile_bid, av_profile_ask = lobreader.average_profile(str(num_levels_calc),write_outputfile=True)
            lobp.plot_av_profile(av_profile_bid, av_profile_ask, filename, ticker_str, date_str, str(time_start), str(time_end))
        
        tvbid = np.sum(av_profile_bid)
        tvask = np.sum(av_profile_ask)
        modelLSQ, modelLSQF, modelArgMax, modelRMax1 = cal.fit_profile_to_data(np.array(av_profile_bid), np.array(av_profile_ask))
        modelLSQ.set_modelid(lobreader.create_filestr("Model-LSQ", str(num_levels_calc)))
        modelLSQF.set_modelid(lobreader.create_filestr("Model-LSQF", str(num_levels_calc)))
        modelArgMax.set_modelid(lobreader.create_filestr("Model-ArgMax", str(num_levels_calc)))
        modelRMax1.set_modelid(lobreader.create_filestr("Model-RMax1", str(num_levels_calc)))
        models= [modelLSQ, modelLSQF, modelArgMax, modelRMax1]
        print("Save model parameters to files")

        for model, gamma_bids, gamma_asks in zip(models, (gamma_bids_LSQ, gamma_bids_LSQF, gamma_bids_ArgMax, gamma_bids_RMax1), (gamma_asks_LSQ, gamma_asks_LSQF, gamma_asks_ArgMax, gamma_asks_RMax1)):
            model.savef()
            gb, ga = model.get_gamma()
            gamma_bids.append(gb)
            gamma_asks.append(ga)      

            print("|--------------------------------------------------------\n|")
            print(" Model parameters for modelid %s"%(model.get_modelid()))
            print(" gamma_bid: %f, gamma_ask: %f"%(model.get_gamma()))
            print(" z0_bid: %f, z0_ask: %f"%(model.get_z0()))
            print(" TV_bid: %f, TV_ask: %f"%(tvbid, tvask))
            print("|\n|--------------------------------------------------------")

        lobp.plot_avprofile_fits(av_profile_bid, av_profile_ask, models,labels_leg=["data", "LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"], title_str=('Average Profile\nTicker: {0}, Date: {1}\n Time: {2} to {3}'.format(ticker_str, date_str,time_start,time_end)), filename=lobreader.create_filestr("av-orderbook-fits", str(num_levels_calc)))

    lobreader.set_timecalc(str(time_start_calc), str(time_end_calc))

    filename = lobreader.create_filestr("gamma", str(num_levels_calc))
    gammas = np.array([time_ends, gamma_bids_LSQ, gamma_bids_LSQF, gamma_bids_ArgMax, gamma_bids_RMax1, gamma_asks_LSQ, gamma_asks_LSQF, gamma_asks_ArgMax, gamma_asks_RMax1])
    print(str(gammas.transpose()))
    np.savetxt(
        ".".join((filename, "csv")),
        (gammas.transpose()),
        fmt='%.10f',
        delimiter=',',
        header="Time, gamma_bid LSQ,LSQ,gamma_bid LSQF,gamma_bid ArgMax,gamma_bid RMax1,gamma_ask LSQ,gamma_ask LSQF,gamma_bid ArgMax,gamma_ask RMax1",
        comments=""            
    )
    
    print("Estimators for gamma written......")
    print("Creating plot.")
    title_str = "Estimated $\gamma_b$ and $\gamma_a$\nTicker: {0}, Date: {1}\nStart Time: {2}s, End Time: {3}s".format(ticker_str, date_str, str(int(int(time_start)/1000)), str(int(int(time_end)/1000)))
    
    lobp.plot_avprofile_gamma(filename, time_ends, gammas[1:5,:], gammas[5:,:], labels_leg=["LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"], title_str=title_str)


    #lobp.plot_avprofile_fittings
    
    print("Plots saved.")
    print("Finished.")


def calibrate_profile_lobster_new(
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        time_cal,
        cal_method_profile="LSQF",
):
    """ Splits the day into time intervals of specified size and fits the model profile to the average profile of the order book in the respective periods, reading lobster data. In addition, the average profile for the whole day will be fi
    -----------
    args:
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        cal_method_profile="LSQF",
        time_cal            time for averageing in ms
    """

    gamma_bids = []
    gamma_asks = []

    meanvol_bids = []
    meanvol_aks = []    

    

    lobreader = lobr.LOBSTERReader(
        ticker_str,
        date_str,
        str(time_start_data),
        str(time_end_data),        
        str(num_levels_data),
        str(time_start_calc),
        str(time_end_calc)
    )

    computation_interval = int(time_end_calc) - int(time_start_calc)
    num_calibration = computation_interval / int(time_cal)
    if num_calibration == 0:
        return 0
    
    time_starts = np.arange(int(time_start_calc), int(time_end_calc), int(time_cal))
    time_ends = time_starts + int(time_cal)
    if time_ends[-1] > time_end_calc:
        time_ends[-1] = time_end_calc
    for time_start, time_end in zip(time_starts, time_ends):
        print("Calibrate")        
        lobreader.set_timecalc(str(time_start), str(time_end))
        filename = lobreader.create_filestr(lobr.AV_ORDERBOOK_FILE_ID, str(num_levels_calc))
        
        av_profile_bid, av_profile_ask = lobr.get_data_from_file(filename)
        if (av_profile_bid is None) or (av_profile_ask is None):
            av_profile_bid, av_profile_ask = lobreader.average_profile(str(num_levels_calc),write_outputfile=True)
            lobp.plot_av_profile(av_profile_bid, av_profile_ask, filename, ticker_str, date_str, str(time_start), str(time_end))
        
        tvbid = np.sum(av_profile_bid)
        tvask = np.sum(av_profile_ask)
        modelLSQ, modelLSQF, modelArgMax, modelRMax1 = cal.fit_profile_to_data(np.array(av_profile_bid), np.array(av_profile_ask))
        modelLSQ.set_modelid(lobreader.create_filestr("Model-LSQ", str(num_levels_calc)))
        modelLSQF.set_modelid(lobreader.create_filestr("Model-LSQF", str(num_levels_calc)))
        modelArgMax.set_modelid(lobreader.create_filestr("Model-ArgMax", str(num_levels_calc)))
        modelRMax1.set_modelid(lobreader.create_filestr("Model-RMax1", str(num_levels_calc)))
        models= [modelLSQ, modelLSQF, modelArgMax, modelRMax1]
        print("Save model parameters to files")

        for model, gamma_bids, gamma_asks in zip(models, (gamma_bids_LSQ, gamma_bids_LSQF, gamma_bids_ArgMax, gamma_bids_RMax1), (gamma_asks_LSQ, gamma_asks_LSQF, gamma_asks_ArgMax, gamma_asks_RMax1)):
            model.savef()
            gb, ga = model.get_gamma()
            gamma_bids.append(gb)
            gamma_asks.append(ga)      

            print("|--------------------------------------------------------\n|")
            print(" Model parameters for modelid %s"%(model.get_modelid()))
            print(" gamma_bid: %f, gamma_ask: %f"%(model.get_gamma()))
            print(" z0_bid: %f, z0_ask: %f"%(model.get_z0()))
            print(" TV_bid: %f, TV_ask: %f"%(tvbid, tvask))
            print("|\n|--------------------------------------------------------")

        lobp.plot_avprofile_fits(av_profile_bid, av_profile_ask, models,labels_leg=["data", "LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"], title_str=('Average Profile\nTicker: {0}, Date: {1}\n Time: {2} to {3}'.format(ticker_str, date_str,time_start,time_end)), filename=lobreader.create_filestr("av-orderbook-fits", str(num_levels_calc)))

    lobreader.set_timecalc(str(time_start_calc), str(time_end_calc))

    filename = lobreader.create_filestr("gamma", str(num_levels_calc))
    gammas = np.array([time_ends, gamma_bids_LSQ, gamma_bids_LSQF, gamma_bids_ArgMax, gamma_bids_RMax1, gamma_asks_LSQ, gamma_asks_LSQF, gamma_asks_ArgMax, gamma_asks_RMax1])
    print(str(gammas.transpose()))
    np.savetxt(
        ".".join((filename, "csv")),
        (gammas.transpose()),
        fmt='%.10f',
        delimiter=',',
        header="Time, gamma_bid LSQ,LSQ,gamma_bid LSQF,gamma_bid ArgMax,gamma_bid RMax1,gamma_ask LSQ,gamma_ask LSQF,gamma_bid ArgMax,gamma_ask RMax1",
        comments=""            
    )
    
    print("Estimators for gamma written......")
    print("Creating plot.")
    title_str = "Estimated $\gamma_b$ and $\gamma_a$\nTicker: {0}, Date: {1}\nStart Time: {2}s, End Time: {3}s".format(ticker_str, date_str, str(int(int(time_start)/1000)), str(int(int(time_end)/1000)))
    
    lobp.plot_avprofile_gamma(filename, time_ends, gammas[1:5,:], gammas[5:,:], labels_leg=["LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"], title_str=title_str)


    #lobp.plot_avprofile_fittings
    
    print("Plots saved.")
    print("Finished.")



def extract_volume_lobster(
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        ntimepoints_grid
):
    """ Extract the volume process from lobster and save as csv and plots 
    ----------
    args:
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        ntimepoints_grid       if None, then the process is extracted for all time points in the data, else a uniform grid is created
    
    OUTPUT:
        produces files with identifier: volume
    
    """
    
    # read files from lobster to uniform grid
    lobreader = lobr.LOBSTERReader(
        ticker_str,
        date_str,
        str(time_start_data),
        str(time_end_data),
        str(num_levels_data),
        str(time_start_calc),
        str(time_end_calc)
    )
    
    print('Extracting total volume process.')    
    
    dt, time_stamps, volume_bid, volume_ask = lobreader.load_ordervolume(
        num_observations=ntimepoints_grid,     
        write_output=True
    )
    print('Plotting data')
    title_str="Order volume in first {0} buckets\n ticker: {1}, Date: {2}".format(num_levels_data, ticker_str, date_str)
    filename="_".join((ticker_str, date_str, str(time_start_calc), str(time_end_calc), "ordervolume", str(num_levels_data)))
    lobp.plot_bidaskdata(time_stamps, volume_bid, volume_ask, title_str=title_str, filename=filename)
    print("Finished.")
    

    
def extract_price_lobster(
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        ntimepoints_grid
):
    """ Extract the volume process from lobster and save as csv and plots 
    ----------
    args:
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        ntimepoints_grid       if None, then the process is extracted for all time points in the data, else a uniform grid is created
    
    OUTPUT:
        produces files with identifier: volume
    
    """
    lobreader = lobr.LOBSTERReader(
        ticker_str,
        date_str,
        str(time_start_data),
        str(time_end_data),
        str(num_levels_data),
        str(time_start_calc),
        str(time_end_calc)
    )
    
    print('Extracting total volume process.')    
    dt, time_stamps, prices_bid, prices_ask = reader.load_prices(ntimepoints_grid, write_output=True)

    print('Plotting data')
    title_str="Order volume in first {0} buckets\n ticker: {1}, Date: {2}".format(num_levels_data, ticker_str, date_str)
    filename="_".join((ticker_str, date_str, str(time_start_calc), str(time_end_calc), "best_prices", str(num_levels_data)))
    lobp.plot_bidaskdata(time_stamps, prices_bid, prices_ask, title_str=title_str, filename=filename)
    print("Finished.")
                               



def _prediction_rvar_running_frame(
        time_start,
        time_discr,
        data_price,
        data_bid,
        data_ask,
        num_timepoints_calib,
        num_timepoints_recal=1,
        latex=False
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
    price_volpred_rvar = []
    price_volpred_rcg = []
    rel_err_rcg = []
    rel_err_rvar = []    
    price_vol = []
    
    # Set up model and calibrator using realized variance
    model1= lobprice.PriceModel(modelid="price-model-rcg", tick_size=1/float(100.))
    cal1 = cal.OrderVolumeCalibrator(
        calibratorid="cal-price-model-rcg",
        model=model1
    )
    # Set up model and calibrator using autocorrelation
    model2 = lobprice.PriceModel(modelid="price-model-rvar", tick_size=1/float(100.))
    cal2 = cal.OrderVolumeCalibrator(
        calibratorid="cal-price-model-rvar",
        model=model2,
        estimator_dynamics=None,
        estimator_dyn_corr=est.estimate_vol_gBM
    )

    print("Start calibration in time frame")

    for ctr_now in range(num_timepoints_calib, len(data_bid), num_timepoints_recal):

        # Calibrate 
        # time_now = time_start + (ctr_start + num_timepoints_calib) * time_discr
        # calibrate on frame and set
        par = cal2.calibrate(
            time_start + ctr_now * time_discr,
            time_discr,
            data_bid[ctr_now - num_timepoints_calib:ctr_now:],
            data_ask[ctr_now - num_timepoints_calib:ctr_now:]                    
        )
        par2_bid, par2_ask, rho = cal1.calibrate(
            time_start + ctr_now * time_discr,
            time_discr,
            data_bid[ctr_now - num_timepoints_calib:ctr_now:],
            data_ask[ctr_now - num_timepoints_calib:ctr_now:]                    
        )
        # add correlation to model 1
        # calc price vol via 3 different way
        model1.set_rho(model2.get_rho())
        price_vol1 = est.estimate_vol_rv(
                data_price[ctr_now - num_timepoints_calib:ctr_now:],
                (num_timepoints_calib -1 ) * time_discr
        )
        price_vol2 = model1.get_vol()
        price_vol3 = model2.get_vol()


        price_vol.append(price_vol1)        
        price_volpred_rcg.append(price_vol2)
        price_volpred_rvar.append(price_vol3)
        rel_err_rcg.append(math.fabs(price_vol2 - price_vol1) / float(price_vol1))
        rel_err_rvar.append(math.fabs(price_vol3 - price_vol1) / float(price_vol1))
        
        progress = (ctr_now-num_timepoints_calib) / float(len(data_bid)-num_timepoints_calib)
        sys.stdout.write("\r{0:.1f}%".format(progress*100))
        sys.stdout.flush()
        
    # Save calibration history
    sys.stdout.write("\r{0:.1f}%".format(100))
    sys.stdout.flush()
    print("\n")
    results = cal1.history.to_list() + cal2.history.to_list()
    results.append(['price_volpred_rcg']+  price_volpred_rcg)
    results.append(['price_volpred_rvar'] + price_volpred_rvar)
    results.append(['price_vol'] + price_vol)
    results.append(['rel_error_rcg'] + rel_err_rcg)
    results.append(['rel_error_rvar'] + rel_err_rvar)
   
    return results




def vol_estimation(
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        ntimepoints_grid,
        ntimesteps_cal,
        ntimesteps_nextcal,
        ntimesteps_snapshot=None
):
    """ Predicts the volatility of the price by volatility of the log market depths
    -----------
    args:
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        ntimepoints_grid,
        ntimesteps_cal,
        ntimesteps_nextcal,
        cal_to_average=False    calibration to total volume (if False) in the first buckets or average
        cal_to_average_classic=False    calibration to total volume (if False) in the first buckets or average - by just averaging after extraction

    """
    # Step 1: Load data
    print("Load data.....")
   
    # read files from lobster to uniform grid
    lobreader = lobr.LOBSTERReader(
        ticker_str,
        date_str,
        str(time_start_data),
        str(time_end_data),
        str(num_levels_data),
        str(time_start_calc),
        str(time_end_calc)
    )
                               
    print('Extracting market depth and price processes on uniform grid.')    
    
    dt, time_stamps, depth_bid, depth_ask = lobreader.load_marketdepth(
        num_observations=ntimepoints_grid,
        num_levels_calc_str=str(num_levels_calc),        
        write_output=False
    )
    __, __, prices_bid, prices_ask = lobreader.load_prices(ntimepoints_grid,write_output=False)
    prices_mid = (prices_bid+prices_ask) / float(2)
    
    
    # Step 2: Calculation, returns list of lists
    print("Data loaded. Start calculations......")
    # Create pandas frame from the lists and transpose to column oriented
    #time_discr
    results = _prediction_rvar_running_frame(
        float(time_start_calc)/float(1000),
        dt,
        prices_mid,
        depth_bid,
        depth_ask,
        ntimesteps_cal,
        ntimesteps_nextcal,
        latex=False
    )

    # Step 3: Output
    print("Finished. Saving output.")
    df = pd.DataFrame(results, columns=None, index=None).transpose()
    #df = pd.DataFrame(results, columns=False, index)
    filename=lobr.create_lobster_filename(ticker_str, date_str, str(time_start_calc), str(time_end_calc), "cal-vol-pred", str(num_levels_calc))
    "_".join((ticker_str, date_str, str(time_start_calc), str(time_end_calc), "best-prices", str(num_levels_data)))
    df.to_csv(".".join((filename, "csv")), index=False)
    
    #### 
    # if not (ntimesteps_snapshot is None):
    #     ind_snapshots = range(0, len(results[0]), ntimesteps_snapshot)        
    #     df[ind_snapshots].tolatex(".".join(filename, "tex"))
        
    print("Data saved in files with name: {}.".format(filename))
        
