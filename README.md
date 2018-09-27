# lobpy
LOB Model package. 

This package provides limit order book models introduced in Cont, Mueller (2018), Stochastic PDE models of limit order book dynamics.


In general, lobpy can be run as

> lobpy -cp/-cd args


Calibration to Lobster data

Requires: 
order book file, name format: TICKER_DATE_STARTTIMEINMS_ENDTIMEINMS_oderbook_NUMBEROFLEVELS.csv
message file, name format: TICKER_DATE_STARTTIMEINMS_ENDTIMEINMS_message_NUMBEROFLEVELS.csv

Please see lobsterdata.com for further details of file formatation

The calibration of the model dynamics is started as follows:


> lobpy -cd ticker_str date_str time_start_data time_end_data time_start_calc time_end_calc num_levels_data num_levels_calc timegrid_size timesteps_recal [calibration_to_average_flag(0/1)]

The fit of the average profile: 

> lobpy -cp ticker_str date_str time_start_data time_end_data time_start_calc time_end_calc num_levels_data num_levels_calc num_intervals


Test:
> lobpy -t


E. g. running test data:

> python3 -m lobpy -cd "TEST" "01-06-66" 34200000 36000000 34200000 36000000 10 10 10000 1000 1 1



© 2018 ETH Zurich, Marvin S. Mueller
