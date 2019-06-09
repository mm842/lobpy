"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller


"""


import sys
import lobpy.handler as lobh
#import lobpy.tester as lobt



calibrate_dynamics = False
calibrate_profile = False
vol_estimation = False
testing = False
testing_std = False
data = False

print("Note: This functionality so far only supports LOBSTER files. For process of non-lobster data please use the individual package modules.")

if sys.argv[1] == "-h":
    sys.exit("lobpy option args\n \n ------------\n Options: -cd ..... Calibration of dynamics\n -cp ..... Fit of average profile \n -data ..... Extract volume or price process from data \n \n Calibration of dynamics:\n lobpy -cd ticker_str date_str time_start_data time_end_data time_start_calc time_end_calc num_levels_data num_levels_calc timegrid_size timesteps_recal [calibration_to_average_flag(0/1)]\n ------------\n Fit of average profile:\n lobpy -cp ticker_str date_str time_start_data time_end_data time_start_calc time_end_calc num_levels_data num_levels_calc num_intervals\n Extract volume: lobpy -data volume ticker_str date_str time_start_data time_end_data time_start_calc time_end_calc num_levels_data num_levels_calc timegid_size(int or 'full')\n Extract price process: Extract volume: lobpy -data volume ticker_str date_str time_start_data time_end_data time_start_calc time_end_calc num_levels_data timegid_size(int or 'full')\n ------------\n Test:\n lobpy -t")
elif (sys.argv[1] == "-cp") or (sys.argv[1] == "--calibrate_profile"):
    calibrate_profile = True
elif (sys.argv[1] == "-cd") or (sys.argv[1] == "--calibrate_dynamics"):
    calibrate_dynamics = True
elif sys.argv[1] == "-vol":
    vol_estimation = True
elif sys.argv[1] == "-data":    
    data = True    
elif sys.argv[1] == "-tt":
    testing = True
elif sys.argv[1] == "-t":
    testing_std = True
else:
    sys.exit("Choose valid option for running calibration or help: lobpy [-cp/-cd/-data/-vol/-h]")
        
if calibrate_profile:
    ticker_str = sys.argv[2]
    date_str = sys.argv[3]
    time_start_data = int(sys.argv[4])
    time_end_data = int(sys.argv[5])
    time_start_calc = int(sys.argv[6])
    time_end_calc = int(sys.argv[7])
    num_levels_data = int(sys.argv[8])
    num_levels_calc = int(sys.argv[9])
    time_cal = int(sys.argv[10])

    lobh.calibrate_profile_lobster(
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        time_cal
    )
elif calibrate_dynamics:
    ticker_str = sys.argv[2]
    date_str = sys.argv[3]
    time_start_data = int(sys.argv[4])
    time_end_data = int(sys.argv[5])
    time_start_calc = int(sys.argv[6])
    time_end_calc = int(sys.argv[7])
    num_levels_data = int(sys.argv[8])
    num_levels_calc = int(sys.argv[9])
    timegrid_size = int(sys.argv[10])
    ntimesteps_cal = int(sys.argv[11])
    ntimesteps_nextcal = int(sys.argv[12])
    cal_to_average = False
    try:
        if sys.argv[13] in ("1", 1, "True", "average", "Average"):
            cal_to_average = True
    except IndexError:
        pass;
    
    lobh.calibrate_mrevdynamics_lobster_rf(
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        timegrid_size,
        ntimesteps_cal,
        ntimesteps_nextcal,
        cal_to_average
    )
elif data:
    if sys.argv[2] == "volume":
        ticker_str = sys.argv[3]
        date_str = sys.argv[4]
        time_start_data = int(sys.argv[5])
        time_end_data = int(sys.argv[6])
        time_start_calc = int(sys.argv[7])
        time_end_calc = int(sys.argv[8])
        num_levels_data = int(sys.argv[9])
        num_levels_calc = int(sys.argv[10])
        timegrid_size = sys.argv[11]
        if timegrid_size == "full":
            # Take full data
            timegrid_size = None
        else:
            # Take uniformly sized time grid
            timegrid_size = int(timegrid_size)
        lobh.extract_volume_lobster(
            ticker_str,
            date_str,
            time_start_data,
            time_end_data,
            time_start_calc,
            time_end_calc,
            num_levels_data,
            num_levels_calc,
            timegrid_size
        )

    elif sys.argv[2] == "price":
        ticker_str = sys.argv[3]
        date_str = sys.argv[4]
        time_start_data = int(sys.argv[5])
        time_end_data = int(sys.argv[6])
        time_start_calc = int(sys.argv[7])
        time_end_calc = int(sys.argv[8])
        num_levels_data = int(sys.argv[9])
        timegrid_size = sys.argv[11]
        if timegrid_size == "full":
            # Take full data
            timegrid_size = None
        else:
            # Take uniformly sized time grid
            timegrid_size = int(timegrid_size)
        lobh.extract_price_lobster(
            ticker_str,
            date_str,
            time_start_data,
            time_end_data,
            time_start_calc,
            time_end_calc,
            num_levels_data,
            timegrid_size
        )           
    else:
        print("Unknown option selected. Valid choices are: -data volume and -data price.")
elif vol_estimation:
    ticker_str = sys.argv[2]
    date_str = sys.argv[3]
    time_start_data = int(sys.argv[4])
    time_end_data = int(sys.argv[5])
    time_start_calc = int(sys.argv[6])
    time_end_calc = int(sys.argv[7])
    num_levels_data = int(sys.argv[8])
    num_levels_calc = int(sys.argv[9])
    timegrid_size = int(sys.argv[10])
    ntimesteps_cal = int(sys.argv[11])
    ntimesteps_nextcal = int(sys.argv[12])
    num_snapshots = None
    try:
        num_snapshots = int(sys.argv[13])
    except IndexError:
        pass;    
    lobh.vol_estimation(
        ticker_str,
        date_str,
        time_start_data,
        time_end_data,
        time_start_calc,
        time_end_calc,
        num_levels_data,
        num_levels_calc,
        timegrid_size,
        ntimesteps_cal,
        ntimesteps_nextcal,
        num_snapshots
    )        
elif testing_std:
    print("testing has been disabled")
    #lobt._test_calibration()
elif testing:
    print("testing has been disabled")
    # time_discr = float(sys.argv[2])
    # timegrid_size = int(sys.argv[3])
    # ntimesteps_cal = int(sys.argv[4])
    # ntimesteps_nextcal = int(sys.argv[5])
    # lobt._test_calibration(time_discr, num_tpoints=timegrid_size, ntimesteps_cal=ntimesteps_cal, ntimesteps_nextcal=ntimesteps_nextcal)


print("Done.")
