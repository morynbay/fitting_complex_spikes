
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import calcium
import scipy
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences
import lmfit
from lmfit import Model
from lmfit import Minimizer, Parameters, report_fit
from scipy.integrate import simps
from numpy import trapz
import time

# get the start time
st = time.time()

h5_file_path = '/media/bsinvivo2/team_brandon_working_copy/poste_bs_invivo1/neuron_data/2022_04_01/exp_2022-04-01_0001-0050.h5'
ca_df = calcium.load_suite2p_data_for_wavesurfer_h5(h5_file_path, plane='plane0', df_baseline=0.25,
                                                            force_redo=False, include_unclassified_cells=True,
                                                            save_pickle=False, region='dendrites',
                                                            spike_quantile=0.94)

ca_deltaf = ca_df.loc[ca_df['acquisitionNumbers'] == 1]
# ca_deltaf.plot(x='frameTimestamps', y='cell_0_fluo_chan1')
cal1 = ca_deltaf[['frameTimestamps']]     # select column with time of recording "frameTimestamps"
cal2 = ca_deltaf.filter(like='deltaf_chan1', axis=1)   # select all columns with name "cell_N_deltaf_chan1"
cal3 = ca_deltaf[['frameNumbers']]
data = pd.concat([cal1, cal2], axis=1)
col_num = cal2.shape[1]
timestamp_repeated = pd.concat([cal1]*col_num, ignore_index=True)
frameNumbers = pd.concat([cal3]*col_num, ignore_index=True)
one_column_under_another = cal2.melt()
new_data = pd.concat([frameNumbers, timestamp_repeated, one_column_under_another], axis=1)
tt = new_data[['frameTimestamps']]
t = tt.to_numpy().reshape(-1)
y = new_data['value'].to_numpy().reshape(-1)
peaks, properties = find_peaks(y, height=(0.1, 10), distance=8, prominence=0.2)   # distance - points between peaks
prominences = peak_prominences(y, peaks)[0]
contour_heights = y[peaks] - prominences  #  this is used for vertical lines
all_peaks = pd.DataFrame(peaks)
promi = pd.DataFrame(prominences)
begin_promi = pd.concat([all_peaks, promi], axis=1)
exclude_low_promi = begin_promi[begin_promi.iloc[:, 1] > 0.1]
sweep_excl = exclude_low_promi.iloc[:, 0]
sweep = sweep_excl.reset_index(drop=True)
promi_last_point = promi[:-1].reset_index(drop=True)  # remove last value to fit number of rows with 'end_range'
begin_range = sweep[:-1].reset_index(drop=True)       # remove last value to fit number of rows with 'end_range'
end_range = sweep[1:].reset_index(drop=True) - 5      # remove first value, and subtracted several points before peak
between_points = end_range - begin_range
r_range = pd.concat([begin_range, end_range, between_points, promi_last_point], axis=1)
r_range.columns = ['begin_range', 'end_range', 'between_points', 'prominence']
#   we delete in r_range those rows which have lower than 4 points between peaks
between_peaks = pd.DataFrame(r_range.drop(r_range.index[r_range['between_points'] < 4])).reset_index(drop=True)
r = between_peaks.index.values

hhh = new_data.iloc[::300, :]    # for plotting vertical lines separating each cell data
vert = hhh['frameTimestamps'].index.values

plt.plot(y)
plt.plot(peaks, y[peaks], "*")
plt.vlines(x=peaks, ymin=contour_heights, ymax=y[peaks])
plt.vlines(x=vert, ymin=-0.5, ymax=2.0, ls=":", color="green")  # drawing vertical lines
plt.show()


def exp_func(x, exp_parameters):
    a = exp_parameters[0]
    b = exp_parameters[1]
    c = exp_parameters[2]
    return a * np.exp(-b * x) + c


# define objective function: returns the array to be minimized
def fcn2min(coeff, xn, yn):
    """Exponential Model and subtract data."""
    amp = coeff['amp']
    decay = coeff['decay']
    const = coeff['const']
    model = amp * np.exp(-xn*decay) + const
    return model - yn


# create a set of Parameters (original guess values)
params = Parameters()
params.add('amp', value=0.1, vary=True, min=0.0001)
params.add('decay', value=0.1, vary=True, min=0.01)
params.add('const', value=0.1, vary=True, min=-1)

tau_table = []
# head_cols = [col for col in cal2 if 'cell_' in col]
# param_cloud = pd.DataFrame({'a': [0.8, 1.8, 5, 3], 'b': [0.1, 1.1, 5, 3], 'c': [0.3, 1.3, 5, 3]})
fit_param_table = []

iteration = 0
while iteration <= r.max():   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  pay attention
    xn = t[between_peaks.iat[iteration, 0]:between_peaks.iat[iteration, 1]]
    yn = y[between_peaks.iat[iteration, 0]:between_peaks.iat[iteration, 1]]
    num_of_rows = np.size(xn)
    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(xn, yn), nan_policy='omit')
    minim_result = minner.minimize(method='leastsq')  # (exp_func, params, args=(xn, yn), method='leastsq')
    # calculate final result
    final = yn + minim_result.residual
    tau = 1 / minim_result.params['decay'].value
    tau_table.append(tau)
    fitted_param = [minim_result.params['amp'].value, minim_result.params['decay'].value,
                    minim_result.params['const'].value]
    fit_param_table.append(fitted_param)

    iteration += 1

promi_between_peaks = between_peaks['prominence']
pd_param = pd.DataFrame(fit_param_table)
tau_list = pd.DataFrame(tau_table)
tau_fit_params = pd.concat([tau_list, pd_param], axis=1, ignore_index=True)
tau_fit_params.columns = ['tau1', 'a1', 'b1', 'c1']
number_tau_great_1 = tau_fit_params[abs(tau_fit_params['tau1']) > 1].count()


#  -----------------  second fitting with another guess parameters  ---------------------
# tau_fit_params_1 = tau_fit_params.reset_index(drop=True)[(tau_fit_params['b1'] > 1)]
tau_fit_params_2 = tau_fit_params.reset_index(drop=False)[(tau_fit_params['b1'] <= 1)]
second_fit_indexes = tau_fit_params_2['index'].values
between_peaks_2 = between_peaks.iloc[second_fit_indexes].reset_index(drop=False)
tau_table_2 = []
fit_param_table_2 = []
r2 = between_peaks_2.index.values

# create a set of Parameters
params_2 = Parameters()
params_2.add('amp', value=1.8, vary=True, min=0.0001)
params_2.add('decay', value=1.8, vary=True, min=0.01)
params_2.add('const', value=1.8, vary=True, min=-1)

iteration2 = 0
while iteration2 <= r2.max():
    xn2 = t[between_peaks_2.iat[iteration2, 0]:between_peaks_2.iat[iteration2, 1]]
    yn2 = y[between_peaks_2.iat[iteration2, 0]:between_peaks_2.iat[iteration2, 1]]
    num_of_rows2 = np.size(xn2)

    # do fit, here with leastsq model
    minner_2 = Minimizer(fcn2min, params_2, fcn_args=(xn2, yn2), nan_policy='omit')
    minim_result_2 = minner_2.minimize(method='leastsq')  # (exp_func, params, args=(xn, yn), method='leastsq')
    # calculate final result
    final_2 = yn2 + minim_result_2.residual
    tau_2 = 1 / minim_result_2.params['decay'].value
    tau_table_2.append(tau_2)
    fitted_param_2 = [minim_result_2.params['amp'].value, minim_result_2.params['decay'].value,
                      minim_result_2.params['const'].value]
    fit_param_table_2.append(fitted_param_2)

    iteration2 += 1

pd_param_2 = pd.DataFrame(fit_param_table_2, index=second_fit_indexes)
tau_list_2 = pd.DataFrame(tau_table_2, index=second_fit_indexes)
tau_fit_params_2 = pd.concat([tau_list_2, pd_param_2], axis=1, ignore_index=False)
tau_fit_params_2.columns = ['tau2', 'a2', 'b2', 'c2']
number_tau_great_2 = tau_fit_params_2[abs(tau_fit_params_2['tau2']) > 1].count()

all_taus = pd.concat([tau_fit_params, tau_fit_params_2], axis=1, ignore_index=True)
all_taus.columns = ['tau1', 'a1', 'b1', 'c1', 'tau2', 'a2', 'b2', 'c2']
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

tau_fit_params_0 = tau_fit_params.copy()
tau_fit_params_0.loc[tau_fit_params_0.index[second_fit_indexes]] = tau_fit_params_2[:]
tau_and_prominence = pd.concat([tau_fit_params_0, promi_between_peaks], axis=1)




#  --------------------- for integration areas  ------------------------
limit_start = sweep[:-1].reset_index(drop=True) - 5    # remove last value to fit number of rows with 'end_range' and subtract several points before peaks
limit_end = sweep[1:].reset_index(drop=True) - 5
limit_between = limit_end - limit_start
limit_range = pd.concat([limit_start, limit_end, limit_between], axis=1)
limit_range.columns = ['limit_start', 'limit_end', 'limit_between']
#   we delete in r_range those rows which have lower than 4 points between peaks
limit_between = pd.DataFrame(limit_range.drop(limit_range.index[limit_range['limit_between'] < 4])).reset_index(drop=True)
limit_r = between_peaks.index.values
area_table = []
kk = 0
while kk <= limit_r.max():   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  pay attention
    x_var = t[limit_between.iat[kk, 0]:limit_between.iat[kk, 1]]
    y_var = y[limit_between.iat[kk, 0]:limit_between.iat[kk, 1]]
    area = trapz(abs(y_var))
    area_table.append(area)
    kk += 1

area_list = pd.DataFrame(area_table)

data_tau_area_promi = pd.concat([timestamp_repeated, tau_and_prominence, area_list], axis=1)
data_tau_area_promi.columns = ['time', 'tau1', 'a1', 'b1', 'c1', 'prominence', 'area']

#  ---------------------------------------------------------------------


# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

print('Number of taus greater then 1:', number_tau_great_1)
print('Number of taus greater then 1:', number_tau_great_2)

