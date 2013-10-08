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

NPTSt = 5113  # Train
NPTSp = 1796  # Predict

# Minimal script for gaussian process estimation
class Mesonet(object):
    dtimet = np.recarray((NPTSt,1), dtype={"names": ("time",),
                                           "formats": ("datetime64[D]",)})
    dtimep = np.recarray((NPTSp,1), dtype={"names": ("time",),
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
    

def regressLoop(flux, features, nsplit=10, seed=666):
    alphas = np.logspace(-5, 0, 5, base=10)
    models = []
    for alpha in alphas:
        models.append(linear_model.Ridge(normalize=False, fit_intercept=True, alpha=alpha))
        models.append(linear_model.Lasso(normalize=False, fit_intercept=True, alpha=alpha))
        models.append(linear_model.LassoLars(normalize=False, fit_intercept=True, alpha=alpha))
        #models.append(linear_model.SGDRegressor(penalty="l1", fit_intercept=True, alpha=alpha))
    models.append(ensemble.RandomForestRegressor())
    models.append(ensemble.ExtraTreesRegressor())
    models.append(ensemble.AdaBoostRegressor())
    #models.append(ensemble.GradientBoostingRegressor(loss="lad", n_estimators=10000, max_depth=5))
    models.append(ensemble.GradientBoostingRegressor(loss="lad"))
    models.append(tree.DecisionTreeRegressor())
    models.append(tree.ExtraTreeRegressor())
     
    maeavg = []
    for model in models:
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
        print "AVG MAE : %.1f +/- %.1f" % (np.mean(maes), np.std(maes))
        maeavg.append(np.mean(maes))

    idx   = np.argsort(maeavg)
    model = models[idx[0]]
    print "USING", maeavg[idx[0]], model
    return model.fit(features, flux) # fit all data

def animateGpData(mesonets, feature):
    from mpl_toolkits import basemap
    import matplotlib.animation as animation
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(273, 310)

    # GEFS for debuggging
    if True:
        f = h5py.File("../train/%s_latlon_subset_19940101_20071231.nc" % (feature), "r")
    else:
        f = h5py.File("../test/%s_latlon_subset_20080101_20121130.nc" % (feature), "r")
    glats = []
    glons = []
    gvals = []
    for latidx in range(len(f['lat'])):
        for lonidx in range(len(f['lon'])):
            glats.append(f["lat"][latidx])
            glons.append(f["lon"][lonidx]-360.)
            data = f[fMapper[feature]][:,:,:,latidx,lonidx]
            gvals.append(np.ravel(np.mean(data, axis=1)[:,0]))
    glats   = np.array(glats)
    glons   = np.array(glons)
    gvals   = np.array(gvals)

    # Mesonet coords
    mlats = []
    mlons = []
    for mesonet in mesonets.values():
        mlats.append(mesonet.nlat)
        mlons.append(mesonet.elon)
    mlats   = np.array(mlats)
    mlons   = np.array(mlons)
                       
    class SubplotAnimation(animation.TimedAnimation):
        def __init__(self, mesonets, gvals, npts):
            self.npts = npts
            self.mesonets = mesonets
            self.gvals = gvals

            # initialize animation elements
            fig   = plt.figure()
            sp1   = fig.add_subplot(111)
            m     = basemap.Basemap(projection='stere',
                                    lat_0     = glats.mean(),
                                    lon_0     = glons.mean(), 
                                    llcrnrlat = glats.min(),
                                    urcrnrlat = glats.max(),
                                    llcrnrlon = glons.min(),
                                    urcrnrlon = glons.max(),
                                    rsphere=6371200.,resolution='l',area_thresh=10000,ax=sp1)
            m.drawcoastlines()
            m.drawstates()
            m.drawcountries()
            x, y       = m(mlons, mlats)
            self.scat  = m.scatter(x, y, c=x, marker="s", norm=norm, s=20)
            self.title = fig.suptitle("")
            xg, yg     = m(glons, glats)
            self.scat2 = m.scatter(xg, yg, c=xg, marker="o", norm=norm, s=20)

            animation.TimedAnimation.__init__(self, fig, interval=200, blit=True)

        def _draw_frame(self, framedata):
            tstep = framedata
            print "# Animating step", tstep
            vals    = []
            for mesonet in self.mesonets.values():
                vals.append( mesonet.pdatat[feature][tstep] )
        
            self.title.set_text("%s %s" % (feature, mesonet.dtimet[tstep]))
            self.scat.set_array(np.array(vals))
            self.scat2.set_array(self.gvals[:,tstep])
            self._drawn_artists = [self.scat, self.scat2]

        def new_frame_seq(self):
            return iter(range(self.npts))

    ani = SubplotAnimation(mesonets, gvals, 100)

    #animation.FuncAnimation(fig, animate, frames=100, interval=200, blit=True, fargs=(mesonets, title, scat, gvals, cs))
    ani.save("/tmp/caw.mp4", writer="mencoder")

def plotGpData(mesonets, feature, tstep):
    from mpl_toolkits import basemap

    # Mesonet coords
    mlats = []
    mlons = []
    for mesonet in mesonets.values():
        mlats.append(mesonet.nlat)
        mlons.append(mesonet.elon)
    mlats   = np.array(mlats)
    mlons   = np.array(mlons)

    
    vals    = []
    for mesonet in mesonets.values():
        vals.append( mesonet.pdatat[feature][tstep] )

    import pdb; pdb.set_trace()
    

    fig = plt.figure()
    m = basemap.Basemap(projection='stere',
                        lat_0     = mlats.mean(),
                        lon_0     = mlons.mean(), 
                        llcrnrlat = mlats.min(),
                        urcrnrlat = mlats.max(),
                        llcrnrlon = mlons.min(),
                        urcrnrlon = mlons.max(),
                        rsphere=6371200.,resolution='l',area_thresh=10000)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    x, y = m(mlons, mlats)
    m.scatter(x, y, c=vals, marker="s")
    cbar = m.colorbar(location="bottom",pad="5%")
    cbar.set_label(feature)
    fig.suptitle(mesonet.dtimet[tstep])

if __name__ == "__main__":

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

    # Fill in the times
    datafile = "regress.pickle"
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
    if True:
        for mesonet in mesonets.values():
            mesonet.setAstro(mesonet.dtimet, mesonet.datat)
            mesonet.setAstro(mesonet.dtimep, mesonet.datap)


    # Load previous interpolations of prediction data
    buff = open("gp_test.pickle", "rb")
    gps = cPickle.load(buff)
    buff.close()
    stride = 5
    npred  = 3
    domean = False
    for key in mesonets.keys():
        mesonets[key].pdatap = gps[key].pdata[npred::stride,0] # ,0 for mean  ,1 for std
    if domean:
        for key1 in mesonets.keys():
            for key2 in mesonets[key].pdatap.dtype.names:
                for i in range(len(mesonets[key].pdatap[key2])):
                    mesonets[key].pdatap[key2][i] = np.mean(gps[key].pdata[key2][i*5:i*5+5,0])

    # Load previous interpolations of training data
    buff = open("gp_train.pickle", "rb")
    gps = cPickle.load(buff)
    buff.close()
    for key in mesonets.keys():
        mesonets[key].pdatat = gps[key].pdata[npred::stride,0]
    if domean:
        for key1 in mesonets.keys():
            for key2 in mesonets[key].pdatat.dtype.names:
                for i in range(len(mesonets[key].pdatat[key2])):
                    mesonets[key].pdatat[key2][i] = np.mean(gps[key].pdata[key2][i*5:i*5+5,0])

    #animateGpData(mesonets, "tmax_2m")
    #plotGpData(mesonets, "tmax_2m", 1)

    # Set up a regression; inputs are (n_examples x n_features)
    # Here, n_examples are each of the time stamps
    # And n_features are all the measurements, plus astro
    # Do this station by station
    models = {}
    dnames  = ["Date"]
    dtypes  = [np.dtype("a8")]
    fmats   = ["%s"]
    for key in sdata["stid"]:
       dnames.append(key)
       dtypes.append(np.float64)
       fmats.append("%.1f")
    outdata = np.recarray((len(Mesonet.dtimep,)), dtype={"names": dnames, "formats": dtypes})
    outdata["Date"] = sdates

    args  = []
    mkeys = sdata["stid"]
    for m in range(len(mkeys)):
        mkey     = mkeys[m]
        mesonet  = mesonets[mkey]
        keys     = mesonet.pdatat.dtype.names
        nkeys    = len(keys)

        featt    = np.empty((NPTSt, nkeys+2))
        for i in range(nkeys):
            featt[:,i]    = mesonet.pdatat[keys[i]] #[npred::stride,0] # start at npred, stride is npts/day, use first element=mean
        featt[:,nkeys]    = mesonet.datat["sun_alt"]
        featt[:,nkeys+1]  = mesonet.datat["moon_phase"]
        fluxt             = mesonet.datat["flux"]
        #model             = regress(fluxt, featt)

        args.append((featt, fluxt))


    ###

    if not os.path.isfile(datafile):
        results = pool.map(regress, args)

        buff = open(datafile, "wb")
        cPickle.dump(results, buff)
        buff.close()
    else:
        print "# LOADING"
        buff = open(datafile, "rb")
        results = cPickle.load(buff)
        buff.close()
        import pdb; pdb.set_trace()
        for m in range(len(mkeys)):
            mkey     = mkeys[m]
            mesonet  = mesonets[mkey]
            keys     = mesonet.pdatat.dtype.names
            nkeys    = len(keys)

            featp    = np.empty((NPTSp, nkeys+2))
            for i in range(nkeys):
                featp[:,i]    = mesonet.pdatap[keys[i]] #[npred::stride,0] 
            featp[:,nkeys]    = mesonet.datap["sun_alt"]
            featp[:,nkeys+1]  = mesonet.datap["moon_phase"]
            fluxp             = results[i].predict(featp)
            outdata[mkey]     = fluxp
 
        np.savetxt("out.txt", outdata, fmt=fmats, delimiter=",")
        print ",".join(outdata.dtype.names)
