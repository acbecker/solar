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
from gbr_regress_best7 import Mesonet, build_XY, build_mesonets

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
    lags = np.arange(0, len(pcf))
    plt.vlines(lags, 0.0, pcf, lw=6)
    plt.plot(plt.xlim(), [0, 0], 'k')
    plt.xlabel('Lag [days]')
    plt.ylabel('PCF')
    plt.show()
    plt.savefig(base_dir + 'plots/' + mKey + '_resids.png')
    plt.close()

if __name__ == "__main__":

    useAstro = True

    print "Loading Gaussian Process interpolates..."
    trainFile = "gp2_train_constant.pickle"

    buff = open(base_dir + trainFile, "rb")
    train = cPickle.load(buff)
    buff.close()

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

    print 'Building data for each Mesonet...'
    mesonets = build_mesonets()

    # predict value using regression for each mesonet
    pfile = open(base_dir + 'data/all_mesonets_gbr_gp2_features.pickle', 'rb')
    gbr_all = cPickle.load(pfile)
    pfile.close()

    weights = np.genfromtxt(base_dir + 'solar/gbr_gp2_weights.csv')

    m_idx = 0
    print 'Making predictions...'
    args = []
    for mKey in mesonets.keys():

        print "%s " % mKey

        featt, fluxt = build_XY(mesonets[mKey], train[mKey], NPTSt, best_features)

        pfile = open(base_dir + 'data/' + mKey + '_gbr__gp2_features.pickle', 'rb')
        gbr = cPickle.load(pfile)
        pfile.close()

        fluxp_single = gbr.predict(featt)
        fluxp_all = gbr_all.predict(featt)
        fluxp = (1.0 - weights[m_idx]) * fluxp_single + weights[m_idx] * fluxp_all

        resid = fluxt - fluxp
        args.append((resid, mKey))
        m_idx += 1

    map(make_pacf, args)
    # pool.map(make_pacf, args)
