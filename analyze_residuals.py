__author__ = 'brandonkelly'

import sys
import os
import numpy as np
import h5py
import multiprocessing
import cPickle
import ephem
import matplotlib.pyplot as plt
import types
from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import train_test_split
from sklearn import metrics, linear_model, tree, ensemble
from statsmodels.tsa.stattools import pacf
from matplotlib.mlab import detrend_mean
from gbr_regress_best7 import build_mesonets

base_dir = os.environ['HOME'] + '/Projects/Kaggle/OK_solar/'

fMapper = {
    "apcp_sfc": "Total_precipitation",
    "dlwrf_sfc": "Downward_Long-Wave_Rad_Flux",
    "dswrf_sfc": "Downward_Short-Wave_Rad_Flux",
    "pres_msl": "Pressure",
    "pwat_eatm": "Precipitable_water",
    "spfh_2m": "Specific_humidity_height_above_ground",
    "tcdc_eatm": "Total_cloud_cover",
    "tcolc_eatm": "Total_Column-Integrated_Condensate",
    "tmax_2m": "Maximum_temperature",
    "tmin_2m": "Minimum_temperature",
    "tmp_2m": "Temperature_height_above_ground",
    "tmp_sfc": "Temperature_surface",
    "ulwrf_sfc": "Upward_Long-Wave_Rad_Flux_surface",
    "ulwrf_tatm": "Upward_Long-Wave_Rad_Flux",
    "uswrf_sfc": "Upward_Short-Wave_Rad_Flux"
}

wKeys = ("apcp_sfc", "pres_msl", "pwat_eatm",
         "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m",
         "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc")

best_features = ["tmax_2m", "pres_msl", "dswrf_sfc", "pwat_eatm", "spfh_2m", "tmp_2m", "sun_alt"]

best_features = ("apcp_sfc", "dlwrf_sfc", "dswrf_sfc", "pres_msl", "pwat_eatm",
                 "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m",
                 "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc", "sun_alt")


NPTSt = 5113  # Train
NPTSp = 1796  # Predict


def make_pacf(args):

    resid, mKey = args

    pcf = pacf(resid, nlags=12, method='ywmle')

    plt.clf()
    plt.subplot(221)
    plt.hist(resid, bins=100, normed=True)
    plt.title(mKey)
    plt.xlabel('GBR Residuals')
    plt.subplot(222)
    plt.plot(resid, '.')
    plt.ylabel('Residual')
    plt.xlabel('Time')
    plt.xlim(0, len(resid))
    plt.subplot(223)
    lags = np.arange(0, len(pcf))
    plt.vlines(lags, 0.0, pcf, lw=6)
    plt.plot(plt.xlim(), [0, 0], 'k')
    plt.xlabel('Lag [days]')
    plt.ylabel('PACF')
    plt.subplot(224)
    pspec, freq = plt.psd(resid, NFFT=1024, detrend=detrend_mean)
    plt.xscale('log')
    plt.xlabel('Frequency [1 / day]')
    plt.ylabel('Power Spectrum')
    plt.tight_layout()
    #plt.show()
    plt.savefig(base_dir + 'solar/plots/' + mKey + '_resids.png')
    plt.close()

    return pcf, pspec, freq


if __name__ == "__main__":

    buff = open(base_dir + 'solar/gp5_constant_resids12.pickle', "rb")
    resids = cPickle.load(buff)
    buff.close()

    nmesonets = 98
    resids = resids.reshape((NPTSt, nmesonets))

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

    mesonets = build_mesonets()

    args = []
    m_idx = 0
    for mKey in mesonets.keys():
        args.append((resids[:, m_idx], mKey))
        m_idx += 1

    print 'Computing PACFs...'

    # results = map(make_pacf, args)
    results = pool.map(make_pacf, args)

    pacf_all = 0.0
    pspec_all = 0.0
    freq = results[0][2]

    for r in results:
        pacf_all += r[0]
        pspec_all += r[1]

    pacf_all /= len(results)
    pspec_all /= len(pspec_all)

    lags = np.arange(0, len(pacf_all))
    plt.clf()
    plt.vlines(lags, 0.0, pacf_all, lw=6)
    plt.xlabel('Lag [days]')
    plt.ylabel('PACF')
    plt.title('Average over all Mesonets')
    plt.savefig(base_dir + 'solar/plots/average_pacf.png')
    plt.close()

    plt.clf()
    plt.loglog(freq, pspec_all, '-.', lw=3)
    plt.xlabel('Frequency [1 / day]')
    plt.ylabel('Power Spectrum')
    plt.title('Average over all Mesonets')
    plt.savefig(base_dir + 'solar/plots/average_psd.png')
    plt.close()