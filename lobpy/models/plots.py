"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller

plots.py 

functions to plot relevant quantities 

"""
import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import numpy as np

import lobpy.models.calibration as cal


def plot_avprofile_gamma(filename, time_stamps, gammas_bid, gammas_ask, labels_leg=["LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"], title_str="Fit of average profile"):
    """ 
    Plots the resuls from fitting average profiles. Supports up to 4 different methods used for calibration.
    ----------
    args:
        filename:       str for filename of the plot
        time_stamps:    time points of fits
        gammas_bid:     fitted bid values of gamma, multiple methods might have been used and stored as gammas_bid[methodindex,timeindex]
        gammas_ask:     fitted ask values of gamma, multiple methods might have been used and stored as gammas_bid[methodindex,timeindex]
        labels_leg=["LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"]:    Labels corresponding to the different methods used for fit
        title_str       string to be used for plotting
    """
    
    f, (ax0, ax1) = plt.subplots(2, sharex=True, sharey=False)

    bid_styles = ['g', 'g--', 'g-.', 'g-*']
    ask_styles = ['r', 'r--', 'r-.', 'r-*']
    num_methods = gammas_bid[:,0].size
    if num_methods > 4:
        warnings.warn("More than 4 methods provided for gamma not supported. Columns > 4 will not be plotted.", RuntimeWarning)
        num_methods = 4

    for ctr in range(num_methods):        
        ax0.plot(time_stamps, gammas_bid[ctr,:], bid_styles[ctr])
        ax1.plot(time_stamps, gammas_ask[ctr,:], ask_styles[ctr])
        
    #ax0.legend(labels_leg, bbox_to_anchor=(1.05, 1))
    ax0.legend(labels_leg)
    ax0.set_title(title_str)

    #ax1.legend(["LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"], bbox_to_anchor=(1.05, 1))
    ax1.legend(labels_leg)
    ax1.tick_params(axis='x', pad=6)
    ax1.set_xlabel('time in msec from midnight')

    plt.tight_layout()        
    plt.savefig(filename + '.pdf') 
    plt.savefig(filename + '.png')







def plot_av_profile(profile_bid, profile_ask, filename, ticker_str, date_str, time_start, time_end):

    num_levels = profile_bid.size
    ind = np.arange(num_levels) # the x locations for the groups
    width = 0.35       # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, profile_bid, width, color='g')
    
    rects2 = ax.bar(ind + width, profile_ask, width, color='r')
    
    # add some text for labels, title and axes ticks
    ax.set_title('Average Profile\nTicker: {0}, Date: {1}\n Time: {2} to {3}'.format(ticker_str, date_str,time_start,time_end))
    ax.set_xticks(ind + width)
    ax.set_xticklabels((ind+1))
    
    lgd = ax.legend((rects1[0], rects2[0]), ('Bid', 'Ask'))

    fig.savefig(filename + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight') 
    fig.savefig(filename + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)
                 
    return True



def plot_calibration_history_volume(cal_history, filename="calhistoryplot", titlestr="", plot_corr=True):
    """ Plots the history of order volume calibration
    ----------
    args:
        cal_history:    CalibrationHistory object
        filename:       filename for the plot
        titlestr:       title for the plot
        plot_corr:      If false, no plot for the correlation will be added
    """
    # Number of plots (each per parameters nu, mu, sigma, (rho,), one for the data, and one for the number of moments
    num_params = len(cal_history.params_bid)
    if (num_params) != len(cal_history.params_ask):
        warnings.warn("Number of keys on bid and ask side are different. Quit plotting.")
        return False

    if plot_corr:
        num_params += 1
        
    time_stamps = np.array(cal_history.time_stamps)

        
    ############################
    #  Plot results
    ############################
    # Three subplots sharing both x/y axes
    fig, axs = plt.subplots(num_params+1, sharex=True, sharey=False, figsize=(9,7), dpi=100)

    axs[0].plot(
        time_stamps,
        cal_history.params_bid['z0'],
        color='g',
        label=r'$D_b$')
    axs[0].plot(
        time_stamps,
        cal_history.params_ask['z0'],
        color='r',
        label=r'$D_a$')
    
    title_str_full = r'Calibrated Parameters'
    title_str_full += '\n'
    title_str_full += titlestr
    axs[0].set_title(title_str_full)
    axs[0].locator_params(nbins=3, axis='y')
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.175, 1.0), borderaxespad=0.)
    axs[0].margins(x=0)

    # Calculate number of moments which are expected
    moments_bid = (np.divide(cal_history.params_bid['nu'], np.power(cal_history.params_bid['sigma'],2)) * 2. + 1).astype(int)
    moments_ask = (np.divide(cal_history.params_ask['nu'], np.power(cal_history.params_ask['sigma'],2)) * 2. + 1).astype(int)

    # highlight when bid or ask side has less or equal 4 moments (convergence results of estimators not available in that case)
    axs[1].plot(time_stamps, 4. * np.ones(len(time_stamps)) , color='b', linestyle='dashdot')
    
    axs[1].plot(time_stamps, moments_bid, color='g', label=r'$k_b^{\max}$')
    axs[1].plot(time_stamps, moments_ask, color='r', label=r'$k_a^{\max}$')
    axs[1].fill_between(time_stamps, np.clip(moments_bid, 0, 4), 4, where=(moments_bid <= 4), facecolor='g', alpha=0.5) #interpolate=True
    axs[1].fill_between(time_stamps, np.clip(moments_ask, 0, 4), 4, where=(moments_ask <= 4), facecolor='r', alpha=0.5)    
    axs[1].set_ylim([1, 10])
    axs[1].locator_params(nbins=3, axis='y')
    lgd =  axs[1].legend(loc='upper right', bbox_to_anchor=(1.175, 1.0), borderaxespad=0.)    
    axs[1].margins(x=0)

    axs[2].plot(time_stamps, cal_history.params_bid['mu'], color='g', label=r'$\mu_b$')
    axs[2].plot(time_stamps, cal_history.params_ask['mu'], color='r', label=r'$\mu_a$')
    axs[2].locator_params(nbins=3, axis='y')
    axs[2].legend(loc='upper right', bbox_to_anchor=(1.175, 1.0), borderaxespad=0.)
    axs[2].margins(x=0)

    axs[3].plot(time_stamps, cal_history.params_bid['nu'], color='g', label=r'$\nu_b$')
    axs[3].plot(time_stamps, cal_history.params_ask['nu'], color='r', label=r'$\nu_a$')
    axs[3].locator_params(nbins=3, axis='y')
    axs[3].legend(loc='upper right', bbox_to_anchor=(1.175, 1.0), borderaxespad=0.)
    axs[3].margins(x=0)
    
    axs[4].plot(time_stamps, cal_history.params_bid['sigma'], color='g', label=r'$\sigma_b$')
    axs[4].plot(time_stamps, cal_history.params_ask['sigma'], color='r', label=r'$\sigma_a$')
    axs[4].locator_params(nbins=3, axis='y')
    axs[4].legend(loc='upper right', bbox_to_anchor=(1.175, 1.0), borderaxespad=0.)
    axs[4].margins(x=0)

    if plot_corr:
        axs[num_params].plot(time_stamps, cal_history.params_correlation, color='b', label=r'$\rho$')
        axs[num_params].legend(loc='upper right', bbox_to_anchor=(1.175, 1.0), borderaxespad=0.)

        axs[num_params].locator_params(nbins=3, axis='y')
        axs[num_params].margins(x=0)
    
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    #f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    axs[num_params].tick_params(axis='x', pad=6)
    axs[num_params].set_xlabel('time in sec from midnight')
    
    fig.savefig(filename + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight') 
    fig.savefig(filename + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)

    return True
    

def plot_bidaskdata(time_grid, data_bid, data_ask, title_str="", filename=""):
    """ Plot the given bid and ask data with a given title string and stores as filename.pdf and filename.png """
    
    fig = plt.figure()
    plt.plot(time_grid, data_bid, 'g')
    plt.plot(time_grid, data_ask, 'r')
    plt.title(title_str)
    plt.legend(["bid", "ask"])
    plt.xlabel('time in sec from midnight')
    plt.tight_layout()
    if not (filename == ""):
        plt.savefig(filename + '.pdf') 
        plt.savefig(filename + '.png')
    plt.close(fig)
    return True





# def plot_av_profile_fit(, title_str, filename):

#     num_levels = profile_bid.size
#     ind = np.arange(num_levels) # the x locations for the groups
#     width = 0.35       # the width of the bars
    
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(ind, profile_bid, width, color='g')
    
#     rects2 = ax.bar(ind + width, profile_ask, width, color='r')
    
#     # add some text for labels, title and axes ticks
#     ax.set_title(title_str)
#     ax.set_xticks(ind + width)
#     ax.set_xticklabels((ind+1))
    
#     lgd = ax.legend((rects1[0], rects2[0]), ('Bid', 'Ask'))

#     fig.savefig(filename + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight') 
#     fig.savefig(filename + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
#     plt.close(fig)
                 
#    return True

def plot_avprofile_fits(profile_bid, profile_ask, models, labels_leg=["data", "LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"], title_str="Fit of average profile", filename="av-profile"):
    """ 
    Plots the resuls from fitting average profiles. Supports up to 4 different methods used for calibration.
    ----------
    args:
        filename:       str for filename of the plot
        time_stamps:    time points of fits
        models:         list of models with calibrated parameters using possibly different methods
        labels_leg=["LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"]:    Labels corresponding to the different methods used for fit
        title_str       string to be used for plotting
    """


    f, (ax0, ax1) = plt.subplots(2, sharex=True, sharey=False)
    
    bid_styles = ['g', 'g--', 'g-.', 'g-*', 'g-o']
    ask_styles = ['r', 'r--', 'r-.', 'r-*', 'r-o']
    num_models = len(models)
    num_levels = len(profile_bid)
    if num_models > 4:
        warnings.warn("More than 4 models not supported for plooting. Only first 4 will be plotted.", RuntimeWarning)
        num_models = 4
        
    xlevels = np.arange(0,num_levels) + 0.5
    ax0.plot(xlevels, profile_bid, bid_styles[0])
    ax1.plot(xlevels, profile_ask, ask_styles[0]) 
    for ctr in range(num_models):
        profile_m_bid = models[ctr].get_profilefct_bid()
        profile_m_ask = models[ctr].get_profilefct_ask()
        ax0.plot(xlevels, profile_m_bid(xlevels), bid_styles[ctr+1])
        ax1.plot(xlevels, profile_m_ask(xlevels), ask_styles[ctr+1]) 
        
        
    #ax0.legend(labels_leg, bbox_to_anchor=(1.05, 1))
    ax0.legend(labels_leg)
    ax0.set_title(title_str)

    #ax1.legend(["LSQ", "LSQF", "ArgMax", "$R_{\infty, 1}$"], bbox_to_anchor=(1.05, 1))
    ax1.legend(labels_leg)
    ax1.tick_params(axis='x', pad=6)
    #ax1.set_xlabel('x')
    
    plt.tight_layout()        
    plt.savefig(filename + '.pdf') 
    plt.savefig(filename + '.png')
