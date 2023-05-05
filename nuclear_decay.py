# -*- coding: utf-8 -*-
"""
Title: Nuclear decay Assignment 

Header: Determines the half-lives and decay constants of both 79Sr and 79Rb
from sets of data

n35409aa 27/11/2022
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
from scipy.optimize import curve_fit

R_MEASURED = 0.0005
S_MEASURED = 0.005
S_SAMPLE = (10**-6) * (6.02214076*10**23)

CON_T = 3600  # convert the values for time to seconds
CON_A = 10**12  # convert the values for activity and its uncertainty to Bq
T_UBOUND = 3600  # upper bound for time in seconds
OUTLIER_BOUND = 1*10**14  # maximum difference allowed


def readcomb():
    """
    Reads in data sets, combines them, removes errors and changes
    units
    Also removes potential outliers within the data sets

    Returns
    -------
    time: t_1 (float)
    activity: a_1 (float)
    uncertainty: u_1 (float)
    """
    try:

        data_1 = np.genfromtxt('Nuclear_data_1.csv',
                               comments='%', delimiter=',')
        data_2 = np.genfromtxt('Nuclear_data_2.csv',
                               comments='%', delimiter=',')
    except OSError as e_1:
        print('Check file name', e_1)
    except NameError as l_1:
        print('Variable is not defined', l_1)

    data_combine = np.vstack((data_1, data_2))  # combines files

    remove_nan = data_combine[~np.isnan(data_combine).any(
        axis=1)]  # detects rows with nan, then produces an array without them

    remove_zero = remove_nan[remove_nan.all(1)]  # removes rows with zeroes

    # creates a model function to filter out outliers
    act_pre = (-R_MEASURED) *\
        (np.exp(-S_MEASURED * (remove_zero[:, 0]*CON_T)) -
         np.exp(-R_MEASURED * (remove_zero[:, 0]*CON_T))) * S_SAMPLE

    outlier_s = np.where(
        np.abs((remove_zero[:, 1] * CON_A) - act_pre) > OUTLIER_BOUND)

    t_1 = np.abs(np.delete(remove_zero[:, 0], outlier_s) * CON_T)
    a_1 = np.abs(np.delete(remove_zero[:, 1], outlier_s) * CON_A)
    u_1 = np.abs(np.delete(remove_zero[:, 2], outlier_s) * CON_A)

    return t_1, a_1, u_1


def decayguess(time, r_2, s_2):
    """
    Creates a function for scipy.optimize.curve_fit to try and match measured
    data to and predict the optimal parameters for decay constants
    """
    # Splits up the equation into three parts and then combines them

    v_1 = (r_2 * s_2) / (r_2 - s_2)
    i_1 = np.exp(-s_2 * time)
    m_1 = np.exp(-r_2 * time)

    return (v_1 * (i_1 - m_1)) * S_SAMPLE


def decaychi(xy_1):
    """
    Creates a chi-squared function to be minimized by scipy.optimize.fmin to
    return/calculate the optimal parameters for the decay constants
    """
    r_1 = xy_1[0]  # varying decay constant for rubidium
    s_1 = xy_1[1]  # varying decay constant for strontium

    # actiity equation split into three parts
    x_1 = ((r_1*s_1) / (r_1 - s_1))
    y_2 = np.exp(-s_1 * t_data)
    z_3 = np.exp(-r_1 * t_data)

    expected = (x_1 * (y_2 - z_3)) * S_SAMPLE
    chisqr = np.sum(((a_data - expected)/u_data)**2)
    return chisqr


def calcval():
    """
    Calculates half life for both elements, also predicts the activity level
    at 90 minutes and the reduced chi-squared

    Returns
    -------
    rub (float) rubidium half life
    stro (float) strontium half life
    act (float) activity at 90 mins
    chi_r (float) reduced chi squared
    """

    rub = (np.log(2)/r_const)/60
    stro = (np.log(2)/s_const)/60
    act_1 = (((r_const * s_const)/(r_const - s_const)) *
             (np.exp(-s_const * 5400) - np.exp(-r_const * 5400)) * S_SAMPLE) / CON_A
    chi_r = min_chi / (np.size(t_data) - 2)

    return rub, stro, act_1, chi_r


def modelfunc():
    """
    generates a best fit function using the best fit parameters
    """
    # activity equation split into three parts
    par_1 = ((r_const*s_const)/(r_const-s_const))
    par_2 = np.exp(-s_const*time_range)
    par_3 = np.exp(-r_const*time_range)
    return (par_1*(par_2 - par_3)) * S_SAMPLE


def meshes():
    """
    Meshes a range of values for the decay constant for rubidium
    lamda_r (float) and strontium lambda_s (float) and returns a meshified
    result
    """
    lambda_r = np.linspace(0.0003, 0.0007, 100)
    lambda_s = np.linspace(0.003, 0.007, 100)
    return np.meshgrid(lambda_r, lambda_s)


def errorcalc():
    """
    Calculates and returns errors on: decay constants (float), half-lives (float)

    """
    percent_err_r = np.sqrt((uncertainty_x/r_const)**2)
    percent_err_s = np.sqrt((uncertainty_y/s_const)**2)
    err_half_r = percent_err_r * t_rub
    err_half_s = percent_err_s * t_str
    return err_half_r, err_half_s


def plotfunc():
    """
    Collects all the plotting information of the contour and regular plots and
    puts it into one function

    Also collects information across the first contour line and puts it into an
    array
    """
    fig = plt.figure()
    a_x = fig.add_subplot(111)
    contour_plot = a_x.contour(mesh_grid[0], mesh_grid[1], mesh_data, 3)
    a_x.clabel(contour_plot, fontsize=8, colors='r')
    a_x.set_title('Contour plot of the decay constants for Rb and Sr')
    a_x.set_xlabel('Rb (s^-1)')
    a_x.set_ylabel('Sr (s^-1)')
    # collects and creates an array out of the values across the first contour
    collect_info = contour_plot.collections[1].get_paths()[0]
    # collects x and y values of the "vertices" in the contour
    arbitrary_array = collect_info.vertices
    y_vals = arbitrary_array[:, 0]
    x_vals = arbitrary_array[:, 1]
    plt.savefig('contour plot of 2 parameters', dpi=600)
    plt.tight_layout()

    fig = plt.figure()
    a_x = fig.add_subplot(111)
    a_x.errorbar(t_data, a_data, yerr=u_data, alpha=0.8, fmt='o',
                 color='c', label='raw data')
    a_x.plot(time_range, modelfunc(), color='k',
             label='Rb:0.00051, Sr:0.00508, Reduced Chi:1.16')
    a_x.set_title(r'$\beta$ - decay of Rubidium-79')
    a_x.grid(dashes=(1, 1))
    a_x.set_xlabel('Time (s)')
    a_x.set_ylabel('Activity (Bq)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Beta minus decay graph', dpi=600)
    return fig, y_vals, x_vals

# MAIN


# extract time, activity and uncertainty data
t_data, a_data, u_data = readcomb()

# extract initial guess parameters
popt, pcov = curve_fit(decayguess, t_data, a_data, sigma=u_data,
                       p0=(0.0001, 0.001))
r_opt, s_opt = popt
err = np.sqrt(np.diag(pcov))  # finds error on the inital decay constant guess

# minimize chi-squared function and extract best fit parameters
res = fmin(decaychi, (0.001, 0.0001),
           full_output=True, disp=False)
s_const = res[0][0]
r_const = res[0][1]
min_chi = res[1]

# extract: rubidium half life, strontium half life, activity at 90 mins and
# reduced chi-squared
t_rub, t_str, act_90, red_chi = calcval()

# create a range of estimates values from approximated parameters
time_range = np.linspace(0, T_UBOUND, 3600)
activity_range = decayguess(time_range, r_opt, s_opt)

mesh_grid = meshes()  # extracts values from mesh grid function
# applies chi-squared function to mesh grid data across every axis
mesh_data = np.apply_along_axis(decaychi, 0, mesh_grid)

PLOT_FUNC, contour_xvals, contour_yvals = plotfunc()

uncertainty_x = np.abs(np.max(contour_xvals) - r_const)
uncertainty_y = np.abs(np.max(contour_yvals) - s_const)

r_half_life_err, s_half_life_err = errorcalc()

plt.show()

print(u"\nRb decay constant guess: {0:.3g} \u00B1 {8:.3g} s^-1\n\
Sr decay constant guess: {1:.3g} \u00B1 {9:.3g} s^-1\n\
Rb decay constant: {2:.3g} \u00B1 {10:.3g} s^-1\n\
Sr decay constant: {3:.3g} \u00B1 {11:.3g} s^-1 \n\
Rb half life: {4:.3g} \u00B1 {12:.3g} mins\n\
Sr half life: {5:.3g} \u00B1 {13:.3g} mins\n\
Reduced Chi-Squared: {6:.2f}\n\
Activity at 90 mins: {7:.3g} TBq"
      .format(r_opt, s_opt, r_const, s_const, t_rub, t_str, red_chi, act_90,
              err[0], err[1], uncertainty_x, uncertainty_y, r_half_life_err, s_half_life_err))
