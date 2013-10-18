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
import regress_bk
from regress_bk import Mesonet

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

fKeys = ("apcp_sfc", "dlwrf_sfc", "dswrf_sfc", "pres_msl", "pwat_eatm",
         "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m",
         "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc")

wKeys = ("apcp_sfc", "pres_msl", "pwat_eatm",
         "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m",
         "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc")

NPTSt = 5113  # Train
NPTSp = 1796  # Predict

# now save results and make feature importance and partial dependency plots
feature_labels = np.array(fMapper.values())
useAstro = True
if useAstro:
    feature_labels = np.append(feature_labels, "sun_alt")

print 'Building Mesonets...'
mesonets = regress_bk.build_mesonets()


print "Loading Gaussian Process interpolates..."

trainFile = "gp2_train_constant.pickle"
buff = open(base_dir + trainFile, "rb")
train = cPickle.load(buff)
buff.close()

print 'Making plots...'

fimportance_mean = 0.0

for mKey in mesonets.keys():

    print "%s " % mKey

    # build feature and arrays
    featt, fluxt = regress_bk.build_XY(mesonets[mKey], train[mKey], NPTSt, useAstro)

    # grab pickles GBRs
    pfile = open(base_dir + 'data/' + mKey + '_gbr.pickle', 'rb')
    gbr = cPickle.load(pfile)
    pfile.close()

    fimportance = gbr.feature_importances_
    fimportance = fimportance / fimportance.max()
    fimportance_mean += fimportance / len(mesonets)
    sorted_idx = np.argsort(fimportance)
    pos = np.arange(sorted_idx.size) + 0.5
    plt.barh(pos, fimportance[sorted_idx], align='center')
    plt.yticks(pos, feature_labels[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Importance of Variability Features: Gradient Boosted Regression")
    plt.tight_layout()
    plt.savefig(base_dir + 'solar/plots/' + mKey + '_feature_importance_gbr.png')
    plt.close()

    f_idx = 0
    for f in feature_labels:
        plt.clf()
        ensemble.partial_dependence.plot_partial_dependence(gbr, featt, [f_idx])
        plt.xlabel(f)
        plt.title(mKey)
        plt.savefig(base_dir + 'solar/plots/' + mKey + '_gbr_partial_' + f + '.png')
        plt.close()
        f_idx += 1

sorted_idx = np.argsort(fimportance_mean)
pos = np.arange(sorted_idx.size) + 0.5
plt.barh(pos, fimportance_mean[sorted_idx], align='center')
plt.yticks(pos, feature_labels[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Average Importance of Variability Features")
plt.tight_layout()
plt.savefig(base_dir + 'solar/plots/avg_feature_importance_gbr.png')
plt.close()