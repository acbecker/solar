__author__ = 'brandonkelly'

import os
import numpy as np
import multiprocessing
import cPickle
import ephem
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg
from sklearn import metrics, linear_model, tree, ensemble
from sklearn.decomposition import PCA
# import hmlinmae_gibbs as hmlin

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
    # "tcdc_eatm": "Total_cloud_cover",
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
         "spfh_2m",  # "tcdc_eatm",
         "tcolc_eatm", "tmax_2m", "tmin_2m",
         "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc")

wKeys = ("apcp_sfc", "pres_msl", "pwat_eatm",
         "spfh_2m", "tcdc_eatm",  # "tcolc_eatm"
         "tmax_2m", "tmin_2m",
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
        self.weightst = np.ones((NPTSt, 5))
        self.weightsp = np.ones((NPTSp, 5))
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

    def set_flux_weights(self, time, weights):
        # set weights at each forecast hour, proportional to the intensity of solar radiation as a function
        # of airmass and elevation
        sun = ephem.Sun()
        obs = ephem.Observer()
        obs.lon = (self.elon * np.pi / 180)  # need radians
        obs.lat = (self.nlat * np.pi / 180)  # need radians
        obs.elevation = 0.0 # equation requires airmass at sea level
        fhours = ['12:00:00', '15:00:00', '18:00:00', '21:00:00', '24:00:00']
        for i in range(len(time)):
            fh_idx = 0
            for fh in fhours:
                obs.date = str(time[i]) + ' ' + fh
                sun.compute(obs)
                if sun.alt < 0.0:
                    weights[i, fh_idx] = 0.0
                else:
                    zenith = 90.0 - sun.alt * (180.0 / np.pi)
                    zenith_rad = zenith * np.pi / 180.0
                    airmass = 1.0 / (np.cos(zenith_rad) + 0.50572 * (96.07995 - zenith) ** (-1.6364))
                    weights[i, fh_idx] = (1.0 - self.elev / 7100.0) * (0.7 ** (airmass ** 0.678)) + self.elev / 7100.0
                fh_idx += 1
            weights[i, :] = weights[i, :] / np.sum(weights[i, :])


def boost_regress(X, y):

    gbr = ensemble.GradientBoostingRegressor(loss='lad', max_depth=5, subsample=0.5, learning_rate=0.1,
                                             n_estimators=2000, verbose=1)
    gbr.fit(X, y)

    do_plot = True
    oob_error = -np.cumsum(gbr.oob_improvement_)

    if do_plot:
        plt.plot(oob_error)
        plt.xlabel("# of estimators")
        plt.ylabel("OOB MAE")
        plt.savefig(base_dir + 'solar/plots/oob_error.png')
        # plt.show()
        plt.close()

    ntrees = np.max(np.array([np.argmin(oob_error) + 1, 5]))
    print 'Best model has', ntrees, 'trees.'
    gbr.n_estimators = ntrees
    gbr.fit(X, y)

    if do_plot:
        feature_labels = []
        for f in fKeys:
            feature_labels.append(fMapper[f])
        feature_labels.extend(['airmass', 'latitude', 'longitude', 'elevation'])
        # get rid of original temperature labels
        feature_labels = np.delete(np.array(feature_labels), [7, 8, 9, 10, 11])
        feature_labels = np.insert(feature_labels, 7, 'Temperature PC 1')
        fimportance = gbr.feature_importances_
        fimportance = fimportance / fimportance.max()
        sorted_idx = np.argsort(fimportance)
        pos = np.arange(sorted_idx.size) + 0.5
        plt.clf()
        plt.barh(pos, fimportance[sorted_idx], align='center')
        plt.yticks(pos, feature_labels[sorted_idx])
        plt.xlabel("Relative Importance")
        plt.title("Importance of Variability Features: Gradient Boosted Regression")
        plt.tight_layout()
        plt.savefig(base_dir + 'solar/plots/feature_importance_gbr_resid.png')
        plt.close()

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

    do_weights = False
    if do_weights:
        for mesonet in mesonets.values():
            mesonet.set_flux_weights(mesonet.dtimet, mesonet.weightst)
            mesonet.set_flux_weights(mesonet.dtimep, mesonet.weightsp)

    # Do we do Astro terms?
    print 'Setting Astro terms...'
    useAstro = True
    if useAstro:
        for mesonet in mesonets.values():
            mesonet.setAstro(mesonet.dtimet, mesonet.datat)
            mesonet.setAstro(mesonet.dtimep, mesonet.datap)

    return mesonets


def build_X(mesonets, gptrain, gptest):

    nX = NPTSt

    # Take median over all ensembles, select each hour
    hstride = 5

    m_idx = 0
    for mkey in mesonets.keys():

        print mkey

        this_featt = np.zeros((nX, len(fKeys)))

        # Take median over all ensembles, select each hour
        weights = mesonets[mkey].weightst

        for hKey in range(5):  # which hour

            for f in range(len(fKeys)):
                fKey = fKeys[f]
                # median over ensembles
                feat_h = np.ravel(gptrain[mkey].pdata[fKey])[hKey::hstride]
                # feat_h *= weights[:, hKey]
                this_featt[:, f] += feat_h

        if m_idx == 0:
            featt = this_featt
        else:
            featt = np.vstack((featt, this_featt))

        # add log(airmass) to feature set
        zenith = 90.0 - mesonets[mkey].datat["sun_alt"]
        airmass = 1.0 / (np.cos(zenith * np.pi / 180.0) + 0.50572 * (96.07995 - zenith) ** (-1.6364))

        this_site = np.ones((NPTSt, 3))
        this_site[:, 0] = mesonets[mkey].nlat * np.random.uniform(0.99, 1.01, NPTSt)
        this_site[:, 1] = mesonets[mkey].elon * np.random.uniform(0.99, 1.01, NPTSt)
        this_site[:, 2] = mesonets[mkey].elev * np.random.uniform(0.99, 1.01, NPTSt)

        if m_idx == 0:
            log_airmass = np.log10(airmass)
            site_data = this_site
        else:
            log_airmass = np.hstack((log_airmass, np.log10(airmass)))
            site_data = np.vstack((site_data, this_site))


        m_idx += 1

    # clean and transform data
    for f in xrange(len(fKeys)):
        zeros = np.where(featt[:, f] == 0)[0]
        featt[zeros, f] = np.min(np.abs(featt[featt[:, f] != 0, f]))
        featt[zeros, f] *= np.random.uniform(0.97, 1.03, len(zeros))
        featt[:, f] = np.log10(np.abs(featt[:, f]))

    # use PCA to construct new temperature feature
    pca = PCA(n_components=1)
    pca.fit(featt[:, 7:12])
    tfeature = np.squeeze(pca.transform(featt[:, 7:12]))

    X = np.zeros((featt.shape[0], len(fKeys) + 1 - 4))
    X[:, :7] = featt[:, :7]
    X[:, 7] = tfeature
    X[:, 8:-1] = featt[:, 12:]
    X[:, -1] = log_airmass

    do_sites = True
    if do_sites:
        X = np.append(X, site_data, axis=1)

    Xtest = []
    # return values at each mesonet site for test set as a list
    for mkey in mesonets.keys():

        print mkey
        this_featt = np.zeros((NPTSp, len(fKeys)))

        for hkey in range(5):
            for f in range(len(fKeys)):
                fkey = fKeys[f]
                feat_h = np.ravel(gptest[mkey].pdata[fkey])[hkey::hstride]
                # feat_h *= weights[:, hKey]
                this_featt[:, f] += feat_h

        # clean and transform data
        for f in xrange(len(fKeys)):
            zeros = np.where(this_featt[:, f] == 0)[0]
            this_featt[zeros, f] = np.min(np.abs(this_featt[this_featt[:, f] != 0, f]))
            this_featt[zeros, f] *= np.random.uniform(0.97, 1.03, len(zeros))
            this_featt[:, f] = np.log10(np.abs(this_featt[:, f]))

        thisX = np.zeros((NPTSp, len(fKeys) + 1 - 4))
        thisX[:, :7] = this_featt[:, :7]
        thisX[:, 7] = np.squeeze(pca.transform(this_featt[:, 7:12]))
        thisX[:, 8:-1] = this_featt[:, 12:]

        zenith = 90.0 - mesonets[mkey].datap['sun_alt']
        airmass = 1.0 / (np.cos(zenith * np.pi / 180.0) + 0.50572 * (96.07995 - zenith) ** (-1.6364))
        thisX[:, -1] = np.log10(airmass)

        this_site = np.ones((NPTSp, 3))
        this_site[:, 0] = mesonets[mkey].nlat * np.random.uniform(0.99, 1.01, NPTSp)
        this_site[:, 1] = mesonets[mkey].elon * np.random.uniform(0.99, 1.01, NPTSp)
        this_site[:, 2] = mesonets[mkey].elev * np.random.uniform(0.99, 1.01, NPTSp)
        do_sites = True
        if do_sites:
            thisX = np.append(thisX, this_site, axis=1)

        Xtest.append(thisX)

    return X, Xtest


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
    np.savetxt(os.environ['HOME'] + '/Dropbox/gp5_cnst_gbr_all_reduced.csv', outdata, fmt=fmats, delimiter=',')
    # np.savetxt(base_dir + 'solar/gp5_constant_roblinreg_boosted_residuals.csv', outdata, fmt=fmats, delimiter=',')
    print ",".join(outdata.dtype.names)


if __name__ == "__main__":

    useAstro = True

    print "Loading Gaussian Process interpolates..."
    trainFile = "gp5_train_constant.pickle"
    testFile = "gp5_test_constant.pickle"
    buff = open(base_dir + trainFile, "rb")
    gptrain = cPickle.load(buff)
    buff.close()

    buff = open(base_dir + testFile, "rb")
    gptest = cPickle.load(buff)
    buff.close()

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

    print 'Building data for each Mesonet...'
    mesonets = build_mesonets()

    print 'Building feature and response arrays...'

    X, Xtest = build_X(mesonets, gptrain, gptest)
    del gptrain, gptest
    assert(np.all(np.isfinite(X)))
    for Xt in Xtest:
        assert(np.all(np.isfinite(Xt)))

    m_idx = 0
    for mKey in mesonets.keys():

        thisy = mesonets[mKey].datat["flux"]

        assert(np.all(np.isfinite(thisy)))

        if m_idx == 0:
            y = thisy
        else:
            y = np.hstack((y, thisy))

        m_idx += 1

    valerr = 0.0
    idx = 0

    print 'Computing Gradient Boosting Fit...'
    gbr = boost_regress(X, y)

    # now predict the flux values
    fluxp = np.zeros((NPTSp, len(mesonets)))
    m_idx = 0
    print 'Making predictions...'
    for mKey in mesonets.keys():

        print "%s " % mKey

        fluxp[:, m_idx] = gbr.predict(Xtest[m_idx])
        m_idx += 1

    print 'Building submission file...'
    build_submission_file(fluxp, mesonets)

    do_plot = False
    if do_plot:
        # plot values as a sanity check
        m_idx = 0
        for mKey in mesonets.keys():
            plt.clf()
            plt.plot(mesonets[mKey].datat['flux'], 'b.', ms=2)
            plt.plot(NPTSt + 1 + np.arange(NPTSp), fluxp[:, m_idx], 'r.', ms=2)
            plt.title(mKey)
            plt.show()
            plt.close()
            m_idx += 1