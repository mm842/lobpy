"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller
"""

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True



import lobpy.datareader.orderbook as lob

import lobpy.datareader.orderbook as lobr

import lobpy.models.loblinear as lobm
import lobpy.models.loblineartools as lobt
import lobpy.models.plots as lobp
import lobpy.models.calibration as cal
import lobpy.models.estimators as est



def _test_calibration_wocorr(time_discr=0.01, num_tpoints=10001, ntimesteps_cal=1001, ntimesteps_nextcal=2):
    """ This function tests the calibration algorithm by sampling data from reciprocal gamma diffusions for given parameters and then calibrating to the sampled time series """

    tester_bid = lobt.LOBFactorProcess('bid', z0=5000, mu=3500, nu=2.1, sigma=0.8)
    tester_ask = lobt.LOBFactorProcess('ask', z0=4000, mu=4600, nu=2.4, sigma=0.6)    
    print("Sample path")
    data_test_bid = tester_bid.simulate(time_discr, num_tpoints)
    data_test_ask = tester_ask.simulate(time_discr, num_tpoints)    
    print('Start calibration on time frame')
    # Create calibrator object with id inherited from lobster notation and estimator for correlation based on realized covariance
    ov_cal = cal.OrderVolumeCalibrator(
        calibratorid="test_calibration",
        estimator_dynamics=est.estimate_recgamma_diff,
        estimator_corr=est.estimate_gBM_logcorr_rvar        
    )
    
    ov_cal.calibrate_running_frame(
        0.,
        time_discr,
        data_test_bid,
        data_test_ask,
        ntimesteps_cal,
        ntimesteps_nextcal
    )
    # save history as csv file
    print('Calibration finished. Saving csv file.')
    ov_cal.savef_history(csv=True)
    print('Calibration history saved.')
    
    # create plots


    lobp.plot_calibration_history_volume(ov_cal.history, filename=ov_cal.calibratorid, titlestr="Simulation")
    print('Plots saved')


def _test_calibration(time_discr=0.001, num_tpoints=100001, ntimesteps_cal=10001, ntimesteps_nextcal=2):
    """ This function tests the calibration algorithm by sampling data from reciprocal gamma diffusions for given parameters and then calibrating to the sampled time series """

    model = lobm.LOBLinearTwoFactor()
    model.set_nu((4, 2.8))
    model.set_mu((3600, 4500))
    model.set_sigma((.5, 1.1))
    model.set_z0((4000, 3800))
    model.set_rho(-0.1)
    print("Sample path")
    data_test_bid, data_test_ask = model._simulate(time_discr, num_tpoints)
    print('Start calibration on time frame')
    # Create calibrator object with id inherited from lobster notation and estimator for correlation based on realized covariance
    ov_cal = cal.OrderVolumeCalibrator(
        calibratorid="test_calibration",
        estimator_dynamics=est.estimate_recgamma_diff,
        estimator_corr=est.estimate_gBM_logcorr_rvar        
    )
    
    ov_cal.calibrate_running_frame(
        0.,
        time_discr,
        data_test_bid,
        data_test_ask,
        ntimesteps_cal,
        ntimesteps_nextcal
    )
    # save history as csv file
    print('Calibration finished. Saving csv file.')
    ov_cal.savef_history(csv=True)
    print('Calibration history saved.')
    
    # create plots


    lobp.plot_calibration_history_volume(ov_cal.history, filename=ov_cal.calibratorid, titlestr="Simulation")
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
        time_cal,
        time_recal
):

    gamma_bids_LSQ = []
    gamma_asks_LSQ = []
    
    gamma_bids_LSQF = []
    gamma_asks_LSQF = []
    
    gamma_bids_ArgMax = []
    gamma_asks_ArgMax = []
    
    gamma_bids_RMax1 = []
    gamma_asks_RMax1 = []
    

    lobreader = lobd.LOBSTERReader(
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
    
    time_starts = np.arange(int(time_start_cal), int(time_end_cal), int(time_cal))
    time_ends = time_start + int(time_start_cal)
    
    for time_start, time_end in zip(time_starts, time_ends):
        lobreader.set_timecalc(str(time_start), str(time_end))
        filename = lobreader.create_filestr(lobd.AV_ORDERBOOK_FILE_ID, str(num_levels_calc))
        
        av_profile_bid, av_profile_ask = lobd.get_data_from_file(filename)
        if (av_profile_bid is None) or (av_profile_ask is None):
            av_profile_bid, av_profile_ask = lobreader.average_profile(str(num_levels_calc),write_outputfile=True)
            lobp.plot_av_profile(av_profile_bid, av_profile_ask, filename, ticker_str, date_str, str(time_start), str(time_end))
        
        tvbid = np.sum(av_profile_bid)
        tvask = np.sum(av_profile_ask)
        modelLSQ, modelLSQF, modelArgMax, modelRMax1 = lobc.fit_profile_to_data(np.array(av_profile_bid), np.array(av_profile_ask))
        modelLSQ.set_modelid(lobreader.create_filestr("Model-LSQ", str(num_levels_calc)))
        modelLSQF.set_modelid(lobreader.create_filestr("Model-LSQF", str(num_levels_calc)))
        modelArgMax.set_modelid(lobreader.create_filestr("Model-ArgMax", str(num_levels_calc)))
        modelRMax1.set_modelid(lobreader.create_filestr("Model-RMax1", str(num_levels_calc)))
        print("Save model parameters to files")
        for model, gamma_bids, gamma_asks in zip((modelLSQ, modelLSQF, modelArgMax, modelRMax1), (gamma_bids_LSQ, gamma_bids_LSQF, gamma_bids_ArgMax, gamma_bids_RMax1), (gamma_asks_LSQ, gamma_asks_LSQF, gamma_asks_ArgMax, gamma_asks_RMax1)):
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
            

        lobreader.set_timecalc(str(time_start_calc), str(time_end_calc))
        filename = lobreader.create_filestr("gamma", str(num_levels_calc))
        gammas = np.array([time_end_points, gamma_bids_LSQ, gamma_asks_LSQ, gamma_bids_LSQF, gamma_asks_LSQF, gamma_bids_ArgMax, gamma_asks_ArgMax, gamma_asks_RMax1, gamma_asks_RMax1])
        gammas = gammas.transpose()
        np.savetxt(
            ".".join((filename, "csv")),
            gammas,
            fmt='%.10f',
            delimiter=',',
            header="Time,gamma_bid LSQ,gamma_bid LSQF,gamma_bid ArgMax,gamma_bid RMax1,gamma_ask LSQ,gamma_ask LSQF,gamma_bid ArgMax,gamma_ask RMax1",
            comments=""
        )

        print("Estimators for gamma written......")
        print("Creating plot.")
        title_str = "Estimated $\gamma_b$ and $\gamma_a$\nTicker: {0}, Date: {1}\nStart Time: {2}s, End Time: {3}s".format(ticker_str, date_str, str(int(int(time_start)/1000)), str(int(int(time_end)/1000)))
        
        lobp.plot_avprofile_fits(filname, time_stamps, gammas_bid, gammas_ask, labels_leg=["LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"], title_str=title_str)
        
        print("Plots saved.")
        print("Finished.")
