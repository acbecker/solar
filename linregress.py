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
from statsmodels.regression.quantile_regression import QuantReg
from sklearn import metrics, linear_model, tree, ensemble


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

fKeys = ("apcp_sfc", "dlwrf_sfc", "dswrf_sfc", "pres_msl", "pwat_eatm",
         "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m",
         "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc")

wKeys = ("apcp_sfc", "pres_msl", "pwat_eatm",
         "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m",
         "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc")

NPTSt = 5113  # Train
NPTSp = 1796  # Predict


# Minimal script for gaussian process estimation
class Mesonet(object):
    dtimet = np.recarray((NPTSt,), dtype={"names": ("time",),
                                          "formats": ("datetime64[D]",)})
    dtimep = np.recarray((NPTSp,), dtype={"names": ("time",),
                                          "formats": ("datetime64[D]",)})

    def __init__(self, stid, nlat, elon, elev):
        self.stid = stid
        self.nlat = nlat
        self.elon = elon
        self.elev = elev
        self.weights = np.zeros((NPTSt, 5))
        # Measured data
        self.datat = np.recarray((NPTSt,), dtype={"names": ("flux", "sun_alt", "moon_phase"),
                                                  "formats": (np.int64, np.float64, np.float64)})
        self.datap = np.recarray((NPTSp,), dtype={"names": ("flux", "sun_alt", "moon_phase"),
                                                  "formats": (np.int64, np.float64, np.float64)})

    def setAstro(self, time, data):
        sun = ephem.Sun()
        moon = ephem.Moon()
        obs = ephem.Observer()
        obs.lon = (self.elon * np.pi / 180)  # need radians
        obs.lat = (self.nlat * np.pi / 180)  # need radians
        obs.elevation = self.elev                  # meters
        for i in range(len(time)):
            obs.date = str(time[i])
            sun.compute(obs)
            moon.compute(obs)
            data["sun_alt"][i] = float(180 / np.pi * sun.transit_alt)
            data["moon_phase"][i] = moon.moon_phase

    def set_flux_weights(self, time):
        # set weights at each forecast hour, proportional to the intensity of solar radiation as a function
        # of airmass and elevation
        sun = ephem.Sun()
        obs = ephem.Observer()
        obs.lon = (self.elon * np.pi / 180)  # need radians
        obs.lat = (self.nlat * np.pi / 180)  # need radians
        obs.elevation = 0.0 # equation requires airmass at sea level
        fhours = ['12:00:00', '15:00:00', '18:00:00', '21:00:00', '24:00:00']
        self.weights = np.zeros((len(time), len(fhours)))
        for i in range(len(time)):
            fh_idx = 0
            for fh in fhours:
                obs.date = str(time[i]) + ' ' + fh
                sun.compute(obs)
                if sun.alt < 0.0:
                    self.weights[i, fh_idx] = 0.0
                else:
                    zenith = 90.0 - sun.alt * (180.0 / np.pi)
                    zenith_rad = zenith * np.pi / 180.0
                    airmass = 1.0 / (np.cos(zenith_rad) + 0.50572 * (96.07995 - zenith) ** (-1.6364))
                    self.weights[i, fh_idx] = (1.0 - self.elev / 7100.0) * (0.7 ** (airmass ** 0.678)) + self.elev / 7100.0
                fh_idx += 1
            self.weights[i, :] = self.weights[i, :] / np.sum(self.weights[i, :])


def roblin_regress(args):
    features, flux = args
    Xmat = np.column_stack((np.ones(len(flux)), features))
    robreg = QuantReg(flux, Xmat).fit()

    return robreg


def boost_residuals(X, resid):

    gbr = ensemble.GradientBoostingRegressor(loss='lad', max_depth=5, subsample=0.5, learning_rate=0.01,
                                             n_estimators=10000)
    gbr.fit(X, resid)

    oob_error = -np.cumsum(gbr.oob_improvement_)
    plt.plot(oob_error)
    plt.xlabel("# of estimators")
    plt.ylabel("OOB MAE")
    plt.show()

    ntrees = np.max(np.array([np.argmin(oob_error) + 1, 5]))
    gbr.n_estimators = ntrees
    gbr.fit(X, resid)

    return gbr


def build_mesonets():
    # Need to load the positions and times of training data
    sdata = np.loadtxt(base_dir + "station_info.csv", delimiter=",", skiprows=1,
                       dtype=[("stid", np.str_, 4),
                              ("nlat", np.float64),
                              ("elon", np.float64),
                              ("elev", np.float64)])
    fields = np.loadtxt(base_dir + "train/train.csv", skiprows=1, delimiter=",", dtype=np.int64)
    dates = [np.datetime64(str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:8]) for x in fields[:, 0]]
    Mesonet.dtimet = dates
    mesonets = {}
    for sidx in range(len(sdata)):
        s = sdata[sidx]
        station = Mesonet(s[0], s[1], s[2], s[3])
        station.datat["flux"] = fields[:, sidx + 1]
        mesonets[s[0]] = station

    # Dates of prediction data
    fields = np.loadtxt(base_dir + "sampleSubmission.csv", skiprows=1, delimiter=",", unpack=True).astype(np.int)
    dates = [np.datetime64(str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:8]) for x in fields[0]]
    Mesonet.dtimep = dates
    sdates = [np.str(x) for x in fields[0]]

    print 'Computing flux weights...'

    for mesonet in mesonets.values():
        mesonet.set_flux_weights(mesonet.dtimet)

    # Do we do Astro terms?
    useAstro = True
    if useAstro:
        for mesonet in mesonets.values():
            mesonet.setAstro(mesonet.dtimet, mesonet.datat)
            mesonet.setAstro(mesonet.dtimep, mesonet.datap)

    return mesonets


def build_XY(mesonet, gp_interp, nX, useAstro):
    # Take median over all ensembles, select each hour
    hstride = 5
    featt = np.zeros((nX, len(fKeys) + useAstro))

    # Take median over all ensembles, select each hour
    for hKey in range(5):  # which hour

        for f in range(len(fKeys)):
            fKey = fKeys[f]
            # median over ensembles
            feat_h = np.ravel(gp_interp.pdata[fKey])[hKey::hstride]
            feat_h *= mesonet.weights[:, hKey]
            featt[:, f] += feat_h

    if useAstro:
        zenith = 90.0 - mesonet.datat["sun_alt"] * np.pi / 180.0
        airmass = 1.0 / (np.cos(zenith) + 0.50572 * (96.07995 - zenith * 180.0 / np.pi) ** (-1.6364))
        featt[:, len(fKeys)] = np.log10(airmass)

    fluxt = mesonet.datat["flux"]

    return featt, fluxt


if __name__ == "__main__":

    useAstro = True

    print "Loading Gaussian Process interpolates..."
    trainFile = "gp5_train_constant.pickle"

    buff = open(base_dir + trainFile, "rb")
    train = cPickle.load(buff)
    buff.close()

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

    print 'Building data for each Mesonet...'
    mesonets = build_mesonets()

    train_sets = []
    test_sets = []

    for mKey in mesonets.keys():

        print "%s " % mKey

        hstride = 5
        h_idx = 0
        X, y = build_XY(mesonets[mKey], train[mKey], NPTSt, useAstro)

        # TODO: reduce feature space in build_XY
        train_sets.append((X[:3500, :], y[:3500]))
        test_sets.append((X[3500:, :], y[3500:]))

    print 'Doing robust linear regression ...'
    robust_results = pool.map(roblin_regress, train_sets)
    print 'Finished'

    # TODO: need to add in elevation and lat, long data, combine into one big data set

    valerr = 0.0
    idx = 0
    for test, train, robreg in zip(test_sets, train_sets, robust_results):
        ypredict = robreg.predict(np.column_stack((np.ones(test[1].size), test[0])))
        valerr += np.mean(np.abs(ypredict - test[1])) / len(robust_results)
        Xtrain = train[0]
        if idx == 0:
            Xall = Xtrain
            resid = robreg.resid
            Xtest_all = test[0]
            ytest_all = test[1]
        else:
            Xall = np.vstack((Xall, Xtrain))
            resid = np.hstack((resid robreg.resid))
            Xtest_all = np.vstack((Xtest_all, test[0]))
            ytest_all = np.vstack((ytest_all, test[1]))

    print 'Mean error for linear model:', valerr

    print 'Boosting residuals...'
    gbr = boost_residuals(Xall, resid)

    ypredict = gbr.predict(Xtest_all)
    valerr = np.mean(np.abs(ypredict - ytest_all))

    print 'Mean error after boosting residuals:', valerr