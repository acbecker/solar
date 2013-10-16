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


def regress(args):
    features, flux, depth = args
    gbr = ensemble.GradientBoostingRegressor(loss="lad", n_estimators=10000, subsample=0.5, max_depth=depth,
                                             learning_rate=0.01)
    gbr.fit(features, flux)
    oob_error = -np.cumsum(gbr.oob_improvement_)

    do_plot = False
    if do_plot:
        plt.plot(oob_error)
        plt.xlabel('# of trees')
        plt.ylabel('LAD Error Relative to First Model')
        plt.show()

    gbr.n_estimators = oob_error.argmin() + 1
    gbr.fit(features, flux)

    return gbr


if __name__ == "__main__":

    print "Loading Gaussian Process interpolates..."
    trainFile = "gp2_train_constant.pickle"

    buff = open(base_dir + trainFile, "rb")
    train = cPickle.load(buff)
    buff.close()

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

    print 'Loading data for each Mesonet...'

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

    #### first get optimal depth of trees #####
    stride = 11 * 5
    depths = [1, 2, 3, 5, 7, 10]
    validation_errors = np.zeros(len(depths))
    d_idx = 0
    print 'Finding optimal tree depth...'
    for depth in depths:
        # Regress each Mesonet site on its own
        train_args = []
        validate_set = []
        for mKey in mesonets.keys():

            print "%s " % mKey

            hstride = 5
            h_idx = 0
            featt = np.zeros((NPTSt, len(fKeys) + useAstro))

            # Take median over all ensembles, select each hour
            for hKey in range(5):  # which hour

                for f in range(len(fKeys)):
                    fKey = fKeys[f]
                    # median over ensembles
                    feat_h = np.ravel(np.median(train[mKey].pdata[fKey].reshape((NPTSt, 11, 5)), axis=1))[hKey::hstride]
                    feat_h *= mesonets[mKey].weights[:, hKey]
                    featt[:, f] += feat_h

            if useAstro:
                featt[:, len(fKeys)] = mesonets[mKey].datat["sun_alt"]
            fluxt = mesonets[mKey].datat["flux"]

            train_args.append((featt[:3500], fluxt[:3500], depth))
            validate_set.append((featt[:3500], fluxt[:3500]))

        # predict values to get optimal tree depth
        print 'Running GBRs with maximum tree depth of', depth, '...'
        gbrs = pool.map(regress, train_args)
        print 'Finished'

        # gbrs = regress(train_args[0])

        valerr = 0.0
        for gbr, val in zip(gbrs, validate_set):
            fpredict = gbr.predict(val[0])
            valerr += np.mean(np.abs(fpredict - val[1])) / len(gbrs)

        validation_errors[d_idx] = valerr
        d_idx += 1

    print 'Mean error as a function of tree depth:', validation_errors
    best_depth = depths[validation_errors.argmin()]

    del train_args, validate_set  # free memory

    # Now do this fit using all the data
    print "Fitting GBR using all the data..."
    args = []
    for mKey in mesonets.keys():

        print "%s " % mKey

        # Take median over all ensembles, select each hour

        hstride = 5
        h_idx = 0
        featt = np.zeros((NPTSt, len(fKeys) + useAstro))

        # Take median over all ensembles, select each hour
        for hKey in range(5):  # which hour

            for f in range(len(fKeys)):
                fKey = fKeys[f]
                # median over ensembles
                feat_h = np.ravel(np.median(train[mKey].pdata[fKey].reshape((NPTSt, 11, 5)), axis=1))[hKey::hstride]
                feat_h *= mesonets[mKey].weights[:, hKey]
                featt[:, f] += feat_h

        if useAstro:
            featt[:, len(fKeys)] = mesonets[mKey].datat["sun_alt"]
        fluxt = mesonets[mKey].datat["flux"]

        args.append((featt, fluxt, best_depth))

    # run gradient boosting regression
    gbrs = pool.map(regress, args)

    # now save results and make feature importance and partial dependency plots
    feature_labels = np.array(fMapper.values())
    if useAstro:
        feature_labels = np.append(feature_labels, "sun_alt")

    print 'Pickling GBRs and Making plots...'

    for gbr, tset, mKey in zip(gbrs, args, mesonets.keys()):

        print "%s " % mKey

        pfile = open(base_dir + 'data/' + mKey + '_gbr.pickle', 'wb')
        cPickle.dump(gbr, pfile)
        pfile.close()

        fimportance = gbr.feature_importances_
        fimportance = fimportance / fimportance.max()
        sorted_idx = np.argsort(fimportance)
        pos = np.arange(sorted_idx.size) + 0.5
        plt.barh(pos, fimportance[sorted_idx], align='center')
        plt.yticks(pos, feature_labels[sorted_idx])
        plt.xlabel("Relative Importance")
        plt.title("Importance of Variability Features: Gradient Boosted Regression")
        plt.savefig(base_dir + 'plots/' + mKey + '_feature_importance_gbr.png')
        plt.close()

        f_idx = 0
        for f in feature_labels:
            plt.clf()
            ensemble.partial_dependence.plot_partial_dependence(gbr, tset[0], [f_idx])
            plt.xlabel(f)
            plt.title(mKey)
            plt.savefig(base_dir + 'plots/' + mKey + '_gbr_partial_' + f + '.png')
            plt.close()
