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
import os
from gbr_regress_best7 import Mesonet, build_mesonets

# NOTE: endless empehm warnings
# DeprecationWarning: PyOS_ascii_strtod and PyOS_ascii_atof are deprecated.  Use PyOS_string_to_double instead.
# https://github.com/brandon-rhodes/pyephem/issues/18
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

useAstro = True

NPTSt = 5113  # Train
NPTSp = 1796  # Predict


def build_X(mesonet, gp_interp, nX, features):
    # Take median over all ensembles, select each hour
    hstride = 5
    featt = np.zeros((nX, len(features)))

    # Take median over all ensembles, select each hour
    for hKey in range(5):  # which hour

        for f in range(len(features) - 1):
            fKey = features[f]
            # median over ensembles
            feat_h = np.ravel(np.median(gp_interp.pdata[fKey].reshape((nX, 11, 5)), axis=1))[hKey::hstride]
            # feat_h = np.ravel(gp_interp.pdata[fKey].reshape((nX, 5)))[hKey::hstride]
            feat_h *= mesonet.weights[:, hKey]
            featt[:, f] += feat_h

    featt[:, len(features) - 1] = mesonet.datap["sun_alt"]

    return featt


def build_submission_file(fluxp, mesonets):
    # Need to load the positions and times of training data
    sdata = np.loadtxt("../station_info.csv", delimiter=",", skiprows=1,
                       dtype = [("stid", np.str_, 4),
                                ("nlat", np.float64),
                                ("elon", np.float64),
                                ("elev", np.float64)])
    # Dates of prediction data
    fields = np.loadtxt("../sampleSubmission.csv", skiprows=1, delimiter=",", unpack=True).astype(np.int)
    sdates = [np.str(x) for x in fields[0]]

    # Output data
    dnames = ["Date"]
    dtypes = [np.dtype("a8")]
    fmats = ["%s"]
    for key in sdata["stid"]:
       dnames.append(key)
       dtypes.append(np.float64)
       fmats.append("%.1f")
    outdata = np.recarray((len(Mesonet.dtimep,)), dtype={"names": dnames, "formats": dtypes})
    outdata["Date"] = sdates

    for m in range(len(mesonets.keys())):
        outdata[mesonets.keys()[m]] = fluxp[:, m]

    np.savetxt(base_dir + 'solar/gbr_gp2_shrunken_prediction_bk.csv', outdata, fmt=fmats, delimiter=',')
    print ",".join(outdata.dtype.names)


if __name__ == "__main__":

    print "Loading Gaussian Process interpolates..."
    testFile = "gp2_test_constant.pickle"

    buff = open(base_dir + testFile, "rb")
    test = cPickle.load(buff)
    buff.close()

    print 'Building data for each Mesonet...'
    mesonets = build_mesonets()

    for mesonet in mesonets.values():
        mesonet.set_flux_weights(mesonet.dtimep)

    # predict value using regression for each mesonet
    pfile = open(base_dir + 'data/all_mesonets_gbr_gp2_features.pickle', 'rb')
    gbr_all = cPickle.load(pfile)
    pfile.close()

    weights = np.genfromtxt(base_dir + 'solar/gbr_gp2_weights.csv')

    fluxp = np.zeros((NPTSp, len(mesonets)))
    m_idx = 0
    print 'Making predictions...'
    for mKey in mesonets.keys():

        print "%s " % mKey

        featt = build_X(mesonets[mKey], test[mKey], NPTSp, best_features)

        if m_idx == 0:
            featt_stacked = featt
        else:
            featt_stacked = np.vstack((featt_stacked, featt))

        pfile = open(base_dir + 'data/' + mKey + '_gbr__gp2_features.pickle', 'rb')
        gbr = cPickle.load(pfile)
        pfile.close()

        fluxp_single = gbr.predict(featt)
        fluxp_all = gbr_all.predict(featt)
        fluxp[:, m_idx] = (1.0 - weights[m_idx]) * fluxp_single + weights[m_idx] * fluxp_all
        m_idx += 1

    print 'Building submission file...'
    build_submission_file(fluxp, mesonets)

