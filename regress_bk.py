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

# NOTE: endless empehm warnings
# DeprecationWarning: PyOS_ascii_strtod and PyOS_ascii_atof are deprecated.  Use PyOS_string_to_double instead.
# https://github.com/brandon-rhodes/pyephem/issues/18
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

    def set_flux_weights(self, time, elevation):
        sun = ephem.Sun()
        obs = ephem.Observer()
        obs.lon = (self.elon * np.pi / 180)  # need radians
        obs.lat = (self.nlat * np.pi / 180)  # need radians
        obs.elevation = 0.0 # equation requires airmass at sea level
        fhours = ['12:00:00', '15:00:00', '18:00:00', '21:00:00', '24:00:00']
        weights = np.empty((len(time), len(fhours)))
        for i in range(len(time)):
            fh_idx = 0
            for fh in fhours:
                obs.date = str(time[i]) + ' ' + fh
                sun.compute(obs)
                zenith = 90.0 - sun.alt * (180.0 / np.pi)
                zenith_rad = zenith * np.pi / 180.0
                airmass = 1.0 / (np.cos(zenith_rad) + 0.50572 * (96.07995 - zenith) ** (-1.6364))
                weights[i, fh_idx] = (1.0 - elevation / 7100.0) * (0.7 ** (airmass ** 0.678)) + elevation / 7100.0
                if sun.alt < 0.0:
                    # sun is below the horizon, so don't use this model run
                    weights[i, fh_idx] = 0.0
                fh_idx += 1
            weights[i, :] = weights[i, :] / np.sum(weights[i, :])

        return weights

def regress(args):
    features, flux = args
    model = ensemble.GradientBoostingRegressor(loss="lad", n_estimators=1000)
    return model.fit(features, flux)


if __name__ == "__main__":

    suffix = sys.argv[1]
    trainFile = "gp2_train_%s.pickle" % (suffix)
    predFile = "gp2_pred_%s.pickle" % (suffix)

    if suffix.find("logit") > -1:
        buff = open(trainFile, "rb")
        train, fmin, fmax = cPickle.load(buff)
        buff.close()

        buff = open(predFile, "rb")
        pred, fmin, fmax = cPickle.load(buff)
        buff.close()
    else:
        buff = open(trainFile, "rb")
        train = cPickle.load(buff)
        buff.close()

        buff = open(predFile, "rb")
        pred = cPickle.load(buff)
        buff.close()

    # QUESTION: do we logit the flux?  Not sure, might screw up CV interpretation


    #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

    # Need to load the positions and times of training data
    sdata = np.loadtxt("../station_info.csv", delimiter=",", skiprows=1,
                       dtype=[("stid", np.str_, 4),
                              ("nlat", np.float64),
                              ("elon", np.float64),
                              ("elev", np.float64)])
    fields = np.loadtxt("../train.csv", skiprows=1, delimiter=",", dtype=np.int64)
    dates = [np.datetime64(str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:8]) for x in fields[:, 0]]
    Mesonet.dtimet = dates
    mesonets = {}
    for sidx in range(len(sdata)):
        s = sdata[sidx]
        station = Mesonet(s[0], s[1], s[2], s[3])
        station.datat["flux"] = fields[:, sidx + 1]
        mesonets[s[0]] = station

    # Dates of prediction data
    fields = np.loadtxt("../sampleSubmission.csv", skiprows=1, delimiter=",", unpack=True).astype(np.int)
    dates = [np.datetime64(str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:8]) for x in fields[0]]
    Mesonet.dtimep = dates
    sdates = [np.str(x) for x in fields[0]]

    # Do we do Astro terms?
    useAstro = 0
    if useAstro:
        for mesonet in mesonets.values():
            mesonet.setAstro(mesonet.dtimet, mesonet.datat)
            mesonet.setAstro(mesonet.dtimep, mesonet.datap)

    stride = 11 * 5
    # Regress each Mesonet site on its own
    for mKey in mesonets.keys():

        # Look at each ensemble, one by one
        pKey = 0 # which prediction, nelement * nhour
        for eKey in range(11): # which element
            for hKey in range(5): # which hour

                print "%s %d" % (mKey, pKey)

                featt = np.empty((NPTSt, len(fKeys) + 2 * useAstro))
                for f in range(len(fKeys)):
                    fKey = fKeys[f]
                    featt[:, f] = train[mKey].pdata[pKey::stride][fKey]
                if useAstro:
                    featt[:, len(fKeys)] = mesonets[mKey].datat["sun_alt"]
                    featt[:, len(fKeys) + 1] = mesonets[mKey].datat["moon_phase"]
                fluxt = mesonets[mKey].datat["flux"]

                regressLoop(featt, fluxt)
                pKey += 1

        # Now average over all ensembles, select each hour
        hstride = 5
        for hKey in range(5): # which hour

            print "%s %d" % (mKey, pKey)

            featt = np.empty((NPTSt, len(fKeys) + 2 * useAstro))
            for f in range(len(fKeys)):
                fKey = fKeys[f]
                featt[:, f] = np.ravel(np.mean(train[mKey].pdata[fKey].reshape((NPTSt, 11, 5)), axis=1))[hKey::hstride]
            if useAstro:
                featt[:, len(fKeys)] = mesonets[mKey].datat["sun_alt"]
                featt[:, len(fKeys) + 1] = mesonets[mKey].datat["moon_phase"]
            fluxt = mesonets[mKey].datat["flux"]

            regressLoop(featt, fluxt)
            pKey += 1

    # Now regress all sites at once
    stride = 11 * 5
    pKey = 0 # which prediction, nelement * nhour
    for eKey in range(11): # which element
        for hKey in range(5): # which hour

            print "ALL %d" % (pKey)

            featt = np.empty((NPTSt * len(mesonets.keys()), len(fKeys) + 2 * useAstro))
            fluxt = np.empty((NPTSt * len(mesonets.keys())))
            fIdx = 0
            for mKey in mesonets.keys():
                for f in range(len(fKeys)):
                    fKey = fKeys[f]
                    featt[fIdx * NPTSt:(fIdx * NPTSt + NPTSt), f] = train[mKey].pdata[pKey::stride][fKey]
                    #if useAstro:
                #    featt[:,len(fKeys)]    = mesonets[mKey].datat["sun_alt"]
                #    featt[:,len(fKeys)+1]  = mesonets[mKey].datat["moon_phase"]
                fluxt[fIdx * NPTSt:(fIdx * NPTSt + NPTSt)] = mesonets[mKey].datat["flux"]
                fIdx += 1

            regressLoop(featt, fluxt)
            pKey += 1

    # Now average over all ensembles, select each hour
    hstride = 5
    for hKey in range(5): # which hour

        print "ALL %d" % (pKey)
        featt = np.empty((NPTSt * len(mesonets.keys()), len(fKeys) + 2 * useAstro))
        fluxt = np.empty((NPTSt * len(mesonets.keys())))
        fIdx = 0
        for mKey in mesonets.keys():
            for f in range(len(fKeys)):
                fKey = fKeys[f]
                featt[fIdx * NPTSt:(fIdx * NPTSt + NPTSt), f] = \
                    np.ravel(np.mean(train[mKey].pdata[fKey].reshape((NPTSt, 11, 5)), axis=1))[hKey::hstride]
                #if useAstro:
                #    featt[:,len(fKeys)]    = mesonets[mKey].datat["sun_alt"]
                #    featt[:,len(fKeys)+1]  = mesonets[mKey].datat["moon_phase"]
                fluxt[fIdx * NPTSt:(fIdx * NPTSt + NPTSt)] = mesonets[mKey].datat["flux"]

        regressLoop(featt, fluxt)
        pKey += 1
    
