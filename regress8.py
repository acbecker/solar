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

# NOTE, this one does cross-validation of the last 1000 points.
# Instead of a random selections.  Duh...


fMapper = {
    "tmax_2m" : "Maximum_temperature",
    "pres_msl" : "Pressure",
    "dswrf_sfc" : "Downward_Short-Wave_Rad_Flux",
    "pwat_eatm" : "Precipitable_water",
    "spfh_2m" : "Specific_humidity_height_above_ground",
    "tmp_2m" : "Temperature_height_above_ground",
    #
    "tmin_2m" : "Minimum_temperature",
    "ulwrf_sfc" : "Upward_Long-Wave_Rad_Flux_surface",
    "apcp_sfc" : "Total_precipitation",
    "dlwrf_sfc" : "Downward_Long-Wave_Rad_Flux",
    "tcdc_eatm" : "Total_cloud_cover",
    "tcolc_eatm" : "Total_Column-Integrated_Condensate",
    "tmp_sfc" : "Temperature_surface",
    "ulwrf_tatm" : "Upward_Long-Wave_Rad_Flux",
    "uswrf_sfc" : "Upward_Short-Wave_Rad_Flux"
    }

fKeys = ("tmax_2m","pres_msl","dswrf_sfc","pwat_eatm","spfh_2m","tmp_2m")

NPTSt = 5113  # Train
NPTSp = 1796  # Predict

# Minimal script for gaussian process estimation
class Mesonet(object):
    dtimet = np.recarray((NPTSt,), dtype={"names": ("time",),
                                           "formats": ("datetime64[D]",)})
    dtimep = np.recarray((NPTSp,), dtype={"names": ("time",),
                                           "formats": ("datetime64[D]",)})

    def __init__(self, stid, nlat, elon, elev):
        self.stid  = stid
        self.nlat  = nlat
        self.elon  = elon
        self.elev  = elev
        # Measured data
        self.datat = np.recarray((NPTSt,), dtype={"names": ("flux", "sun_alt"),
                                                  "formats": (np.int64, np.float64)})
        self.datap = np.recarray((NPTSp,), dtype={"names": ("flux", "sun_alt"),
                                                  "formats": (np.int64, np.float64)})

    def setAstro(self, time, data):
        sun = ephem.Sun()
        obs = ephem.Observer()
        obs.lon       = (self.elon * np.pi / 180)  # need radians
        obs.lat       = (self.nlat * np.pi / 180)  # need radians
        obs.elevation = self.elev                  # meters

        for i in range(len(time)):
            obs.date = str(time[i])
            sun.compute(obs)

            # LOGIT ASTRO TERMS
            # Sun Alt goes from 0 to 90
            salt   = float(180 / np.pi * sun.transit_alt)
            salt  /= 90.0
            data["sun_alt"][i] = np.log(salt / (1.0 - salt))


def regressTest(feattr, featcv, fluxtr, fluxcv):
    alphas = np.logspace(-5, 1, 6, base=10)
    models = []
    for alpha in alphas:
        models.append(linear_model.Ridge(normalize=True, fit_intercept=True, alpha=alpha))
        models.append(linear_model.Lasso(normalize=True, fit_intercept=True, alpha=alpha))
        models.append(linear_model.LassoLars(normalize=True, fit_intercept=True, alpha=alpha))
    models.append(ensemble.RandomForestRegressor())
    models.append(ensemble.ExtraTreesRegressor())
    models.append(ensemble.AdaBoostRegressor())
    models.append(ensemble.GradientBoostingRegressor(loss="lad", n_estimators=100)) 
    models.append(ensemble.GradientBoostingRegressor(loss="lad", n_estimators=1000))
    models.append(tree.DecisionTreeRegressor())
    models.append(tree.ExtraTreeRegressor())

    maes = []
    for m in range(len(models)):
        model = models[m]
        fit   = model.fit(feattr, fluxtr)
        preds = fit.predict(featcv)
        mae   = metrics.mean_absolute_error(fluxcv, preds)
        print " MAE %d: %.1f" % (m, mae)
        maes.append(mae)

    idx   = np.argsort(maes)
    model = models[idx[0]]
    print "BEST", maes[idx[0]], model
    return model.fit(np.vstack((feattr, featcv)), 
                     np.hstack((fluxtr, fluxcv))
                     ) # fit all data

def sigclip(data, switch):
    mean = np.mean(data, axis=1)
    std  = np.std(data, axis=1)
    idx  = np.where(std == 0.0)
    std[idx] = 1e10
    if switch:
        nsig = np.abs(data  - mean[:,np.newaxis,:]) / std[:,np.newaxis,:]
    else:
        nsig = np.abs(data  - mean[:,np.newaxis]) / std[:,np.newaxis]
    idx  = np.where(nsig > 3.0)
    ma   = np.ma.array(data)
    ma[idx] = np.ma.masked
    return ma.mean(axis=1).data

if __name__ == "__main__":

    suffix     = sys.argv[1]
    useAstro   = 1
    trainFile  = "gp2_train_%s.pickle" % (suffix)
    predFile   = "gp2_pred_%s.pickle" % (suffix)


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

    #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

    # Need to load the positions and times of training data
    sdata = np.loadtxt("../station_info.csv", delimiter=",", skiprows=1, 
                       dtype = [("stid", np.str_, 4), 
                                ("nlat", np.float64),
                                ("elon", np.float64),
                                ("elev", np.float64)])
    fields         = np.loadtxt("../train.csv", skiprows=1, delimiter=",", dtype=np.int64)
    dates          = [np.datetime64(str(x)[:4]+"-"+str(x)[4:6]+"-"+str(x)[6:8]) for x in fields[:,0]]
    Mesonet.dtimet = dates
    mesonets       = {}
    for sidx in range(len(sdata)):
        s                     = sdata[sidx]
        station               = Mesonet(s[0], s[1], s[2], s[3])
        station.datat["flux"] = fields[:,sidx+1]
        mesonets[s[0]]        = station

    # Dates of prediction data
    fields         = np.loadtxt("../sampleSubmission.csv", skiprows=1, delimiter=",", unpack=True).astype(np.int)
    dates          = [np.datetime64(str(x)[:4]+"-"+str(x)[4:6]+"-"+str(x)[6:8]) for x in fields[0]]
    Mesonet.dtimep = dates
    sdates         = [np.str(x) for x in fields[0]]

    # Do we do Astro terms?
    if useAstro:
        for mesonet in mesonets.values():
            mesonet.setAstro(mesonet.dtimet, mesonet.datat)
            mesonet.setAstro(mesonet.dtimep, mesonet.datap)

    nCv      = 1000
    nTr      = NPTSt-nCv
    # Regress each Mesonet site on its own
    for mKey in mesonets.keys():
        print "%s" % (mKey)

        feattr  = np.empty((nTr, len(fKeys) + useAstro))
        featcv  = np.empty((nCv, len(fKeys) + useAstro))
        for f in range(len(fKeys)):
            fKey        = fKeys[f]
            data1       = sigclip(train[mKey].pdata[fKey].reshape((NPTSt, 11, 5)), True)
            data2       = sigclip(data1, False)
            feattr[:,f] = data2[:-nCv]
            featcv[:,f] = data2[-nCv:]
        if useAstro:
            feattr[:,len(fKeys)]    = mesonets[mKey].datat["sun_alt"][:-nCv]
            featcv[:,len(fKeys)]    = mesonets[mKey].datat["sun_alt"][-nCv:]
        fluxtr = mesonets[mKey].datat["flux"][:-nCv]
        fluxcv = mesonets[mKey].datat["flux"][-nCv:]

        regressTest(feattr, featcv, fluxtr, fluxcv)

    ##########3
    ##########3
    ##########3
    ##########3

    # Now regress all sites at once
    print "ALL"
    feattr = np.empty((nTr * len(mesonets.keys()), len(fKeys) + useAstro))
    fluxtr = np.empty((nTr * len(mesonets.keys())))
    featcv = np.empty((nCv * len(mesonets.keys()), len(fKeys) + useAstro))
    fluxcv = np.empty((nCv * len(mesonets.keys())))
    fIdx  = 0
    for mKey in mesonets.keys():
        for f in range(len(fKeys)):
            fKey       = fKeys[f]
            data1      = sigclip(train[mKey].pdata[fKey].reshape((NPTSt, 11, 5)), True)
            data2      = sigclip(data1, False)
            feattr[fIdx*nTr:(fIdx*nTr + nTr),f] = data2[:-nCv]
            featcv[fIdx*nCv:(fIdx*nCv + nCv),f] = data2[-nCv:]

        if useAstro:
            feattr[fIdx*nTr:(fIdx*nTr + nTr),len(fKeys)]    = mesonets[mKey].datat["sun_alt"][:-nCv]
            featcv[fIdx*nCv:(fIdx*nCv + nCv),len(fKeys)]    = mesonets[mKey].datat["sun_alt"][-nCv:]

        fluxtr[fIdx*nTr:(fIdx*nTr + nTr)] = mesonets[mKey].datat["flux"][:-nCv]
        fluxcv[fIdx*nCv:(fIdx*nCv + nCv)] = mesonets[mKey].datat["flux"][-nCv:]
        fIdx += 1

    regressTest(feattr, featcv, fluxtr, fluxcv)
    
