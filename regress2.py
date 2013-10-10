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
    "apcp_sfc" : "Total_precipitation",
    "dlwrf_sfc" : "Downward_Long-Wave_Rad_Flux",
    "dswrf_sfc" : "Downward_Short-Wave_Rad_Flux",
    "pres_msl" : "Pressure",
    "pwat_eatm" : "Precipitable_water",
    "spfh_2m" : "Specific_humidity_height_above_ground",
    "tcdc_eatm" : "Total_cloud_cover",
    "tcolc_eatm" : "Total_Column-Integrated_Condensate",
    "tmax_2m" : "Maximum_temperature",
    "tmin_2m" : "Minimum_temperature",
    "tmp_2m" : "Temperature_height_above_ground",
    "tmp_sfc" : "Temperature_surface",
    "ulwrf_sfc" : "Upward_Long-Wave_Rad_Flux_surface",
    "ulwrf_tatm" : "Upward_Long-Wave_Rad_Flux",
    "uswrf_sfc" : "Upward_Short-Wave_Rad_Flux"
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
        self.stid  = stid
        self.nlat  = nlat
        self.elon  = elon
        self.elev  = elev
        # Measured data
        self.datat = np.recarray((NPTSt,), dtype={"names": ("flux", "sun_alt", "moon_phase"),
                                                  "formats": (np.int64, np.float64, np.float64)})
        self.datap = np.recarray((NPTSp,), dtype={"names": ("flux", "sun_alt", "moon_phase"),
                                                  "formats": (np.int64, np.float64, np.float64)})

    def setAstro(self, time, data):
        sun = ephem.Sun()
        moon = ephem.Moon()
        obs = ephem.Observer()
        obs.lon       = (self.elon * np.pi / 180)  # need radians
        obs.lat       = (self.nlat * np.pi / 180)  # need radians
        obs.elevation = self.elev                  # meters
        for i in range(len(time)):
            obs.date = str(time[i])
            sun.compute(obs)
            moon.compute(obs)
            data["sun_alt"][i] = float(180 / np.pi * sun.transit_alt)
            data["moon_phase"][i] = moon.moon_phase


def regress(args):
    features, flux = args
    model = ensemble.GradientBoostingRegressor(loss="lad", n_estimators=1000)
    return model.fit(features, flux)
    

def regressLoop(features, flux, nsplit=10, seed=666):
    alphas = np.logspace(-5, 1, 6, base=10)
    models = []
    for alpha in alphas:
        models.append(linear_model.Ridge(normalize=True, fit_intercept=True, alpha=alpha))
        models.append(linear_model.Lasso(normalize=True, fit_intercept=True, alpha=alpha))
        models.append(linear_model.LassoLars(normalize=True, fit_intercept=True, alpha=alpha))
    models.append(ensemble.RandomForestRegressor())
    models.append(ensemble.ExtraTreesRegressor())
    models.append(ensemble.AdaBoostRegressor())
    models.append(ensemble.GradientBoostingRegressor(loss="lad", n_estimators=100)) # change to 1000 for better results
    models.append(tree.DecisionTreeRegressor())
    models.append(tree.ExtraTreeRegressor())
     
    maeavg = []
    for m in range(len(models)):
        model = models[m]
        maes = []
        for i in range(nsplit):
            feat_fit, feat_cv, flux_fit, flux_cv = train_test_split(features, flux, test_size=.20, random_state = i*seed)
            try:
               fit = model.fit(feat_fit, flux_fit)
               preds = fit.predict(feat_cv)
               mae = metrics.mean_absolute_error(flux_cv,preds)
               #print "MAE (fold %d/%d): %f" % (i + 1, nsplit, mae)
               maes.append(mae)
            except: 
               continue
        print " AVG MAE %d : %.1f +/- %.1f" % (m, np.mean(maes), np.std(maes))
        maeavg.append(np.mean(maes))

    idx   = np.argsort(maeavg)
    model = models[idx[0]]
    print "BEST", maeavg[idx[0]], model
    return model.fit(features, flux) # fit all data

if __name__ == "__main__":
    
    suffix     = sys.argv[1]
    trainFile  = "gp2_train_%s.pickle" % (suffix)
    predFile   = "gp2_pred_%s.pickle" % (suffix)

    buff = open(trainFile, "rb")
    train = cPickle.load(buff)
    buff.close()

    buff = open(predFile, "rb")
    pred = cPickle.load(buff)
    buff.close()

    # Crap, what to do about Logit transform..?

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

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
    useAstro = 0
    if useAstro:
        for mesonet in mesonets.values():
            mesonet.setAstro(mesonet.dtimet, mesonet.datat)
            mesonet.setAstro(mesonet.dtimep, mesonet.datap)

    stride   = 11 * 5
    # Regress each Mesonet site on its own
    for mKey in mesonets.keys():

        pKey     = 0 # which prediction, nelement * nhour
        for eKey in range(11): # which element
            for hKey in range(5): # which hour

                print "%s %d" % (mKey, pKey)

                featt = np.empty((NPTSt, len(fKeys) + 2 * useAstro))
                for f in range(len(fKeys)):
                    fKey       = fKeys[f]
                    featt[:,f] = train[mKey].pdata[pKey::stride][fKey]
                if useAstro:
                    featt[:,len(fKeys)]    = mesonets[mKey].datat["sun_alt"]
                    featt[:,len(fKeys)+1]  = mesonets[mKey].datat["moon_phase"]
                fluxt = mesonets[mKey].datat["flux"]

                regressLoop(featt, fluxt)
                pKey += 1
                continue



                model = regress(featt, fluxt)

                featp = np.empty((NPTSp, len(fKeys) + 2 * useAstro))
                for f in range(len(fKeys)):
                    fKey       = fKeys[f]
                    featp[:,f] = pred[mKey].pdata[pKey::stride][fKey]
                if useAstro:
                    featp[:,len(fKeys)]    = mesonets[mKey].datap["sun_alt"]
                    featp[:,len(fKeys)+1]  = mesonets[mKey].datap["moon_phase"]
                fluxp  = model.predict(featp)
                
            pKey += 1
