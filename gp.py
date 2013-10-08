import sys
import os
import numpy as np
import h5py
import multiprocessing
import cPickle
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
from sklearn.gaussian_process import GaussianProcess

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

# Minimal script for gaussian process estimation
class Mesonet(object):
    def __init__(self, stid, nlat, elon, elev, npts):
        self.stid = stid
        self.nlat = nlat
        self.elon = elon
        self.elev = elev

        # Gaussian process interpolation; mean and stdev
        self.pdata = np.recarray((5*npts,2), dtype={"names": ("apcp_sfc", "dlwrf_sfc", "dswrf_sfc", "pres_msl", "pwat_eatm", 
                                                              "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m", 
                                                              "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc"),
                                                    "formats": (np.float64, np.float64, np.float64, np.float64, np.float64, 
                                                                np.float64, np.float64, np.float64, np.float64, np.float64, 
                                                                np.float64, np.float64, np.float64, np.float64, np.float64)})

class GEFS(object):
    def __init__(self, stid, nlat, elon, elev, npts):
        self.stid = stid
        self.nlat = nlat
        self.elon = elon
        self.elev = elev

        # Capture mean and stdev
        self.data = np.recarray((5*npts,2), dtype={"names": ("apcp_sfc", "dlwrf_sfc", "dswrf_sfc", "pres_msl", "pwat_eatm", 
                                                             "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m", 
                                                             "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc"),
                                                   "formats": (np.float64, np.float64, np.float64, np.float64, np.float64, 
                                                               np.float64, np.float64, np.float64, np.float64, np.float64, 
                                                               np.float64, np.float64, np.float64, np.float64, np.float64)})

def plotGpData(args, pred):
    from mpl_toolkits import basemap
    vals, dvals, mcoords, gcoords = args
    fig = plt.figure()
    m = basemap.Basemap(projection='stere',
                        lat_0     = gcoords.mean(axis=0)[0],
                        lon_0     = gcoords.mean(axis=0)[1], 
                        llcrnrlat = gcoords.min(axis=0)[0],
                        urcrnrlat = gcoords.max(axis=0)[0],
                        llcrnrlon = gcoords.min(axis=0)[1],
                        urcrnrlon = gcoords.max(axis=0)[1],
                        rsphere=6371200.,resolution='l',area_thresh=10000)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    x, y = m(gcoords[:,1].reshape(9,16), gcoords[:,0].reshape(9,16))
    cs = m.contourf(x, y, vals.reshape(9,16))
    m.scatter(x, y, c=vals, marker="o")
    
    xp, yp = m(mcoords[:,1].reshape(1,1), mcoords[:,0].reshape(1,1))
    m.scatter(xp, yp, c=pred, marker="s")

def runGaussianProcess(args):
    vals, dvals, mcoords, gcoords = args

    nugget = (dvals / vals)**2
    nugget = np.where(np.isfinite(nugget), nugget, 0.05**2)
    nugget = np.where(nugget < 0.05**2, 0.05**2, nugget)
    gp     = GaussianProcess(corr="squared_exponential", 
                             regr="linear",
                             theta0=1e-1, thetaL=1e-2, thetaU=1,
                             normalize=False,
                             nugget=nugget,
                             random_start=1)

    try:
        gpres = gp.fit(gcoords, vals)
        pred, varpred = gp.predict(mcoords, eval_MSE=True)
    except:
        # This is wrong, len(vals) are inputs not outputs.  opefully fixing nugget will avoid this exception
        pred = np.zeros(len(vals)) 
        sigpred = np.zeros(len(vals)) - 1
    else:
        sigpred = np.sqrt(varpred)
    
    return pred, sigpred

if __name__ == "__main__":
    switch   = sys.argv[1]
    if switch == "train":
        npts = 5113
    else:
        npts = 1796

    datafile = "gp_%s.pickle" % (switch)
    if not os.path.isfile(datafile):
        sdata = np.loadtxt("../station_info.csv", delimiter=",", skiprows=1, 
                           dtype = [("stid", np.str_, 4), 
                                    ("nlat", np.float64),
                                    ("elon", np.float64),
                                    ("elev", np.float64)])
        mesonets      = {}
        for sidx in range(len(sdata)): 
            s                    = sdata[sidx]
            station              = Mesonet(s[0], s[1], s[2], s[3], npts)
            mesonets[s[0]]       = station
    
        gefss    = {}
        for key in ["apcp_sfc", "dlwrf_sfc", "dswrf_sfc", "pres_msl", "pwat_eatm", 
                    "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m", 
                    "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc"]:
            print "# LOADING", key
    
            if switch == "train":
                f = h5py.File("../train/%s_latlon_subset_19940101_20071231.nc" % (key), "r")
            else:
                f = h5py.File("../test/%s_latlon_subset_20080101_20121130.nc" % (key), "r")
            
            if len(gefss.keys()) == 0:
                print "# INITIALIZING GEFS"
                sidx = 0
                for latidx in range(len(f['lat'])):
                    for lonidx in range(len(f['lon'])):
                        gefs              = GEFS(sidx, f["lat"][latidx], f["lon"][lonidx]-360., 0.0, npts)
                        gefss[sidx]       = gefs
                        sidx += 1
    
            sidx = 0
            for latidx in range(len(f['lat'])):
                for lonidx in range(len(f['lon'])):
                    gefs = gefss[sidx]
                    data = f[fMapper[key]][:,:,:,latidx,lonidx]
                    gefs.data[key][:,0] = np.ravel(np.mean(data, axis=1))
                    gefs.data[key][:,1] = np.ravel(np.std(data, axis=1))
                    sidx += 1
    
        # Mesonet coords
        mlats = []
        mlons = []
        melevs = []
        for mesonet in mesonets.values():
            mlats.append(mesonet.nlat)
            mlons.append(mesonet.elon)
            melevs.append(mesonet.elev)
        mlats   = np.array(mlats)
        mlons   = np.array(mlons)
        melevs  = np.array(melevs)
        #mcoords = np.array(zip(mlats,mlons,melevs))
        mcoords = np.array(zip(mlats,mlons))

        # GEFS coords
        glats = []
        glons = []
        gelevs = []
        for gefs in gefss.values():
            glats.append(gefs.nlat)
            glons.append(gefs.elon)
            gelevs.append(gefs.elev)
        glats   = np.array(glats)
        glons   = np.array(glons)
        gelevs  = np.array(gelevs)
        #gcoords = np.array(zip(glats,glons,gelevs))
        gcoords = np.array(zip(glats,glons))

        # Get ready to run all the GP
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(int, range(multiprocessing.cpu_count()))  # Trick to "warm up" the Pool

        tsteps = range(gefss.values()[0].data.shape[0])
        ksteps = gefss.values()[0].data.dtype.names
        # debugging
        #tsteps = [14756,]
        #ksteps = ["dswrf_sfc",]
        
        args = []
        for tstep in tsteps:
            print "# PREPPING T", tstep
            for key in ksteps:
                vals  = []
                dvals = []
                for gefs in gefss.values():
                    vals.append(gefs.data[key][tstep,0])
                    dvals.append(gefs.data[key][tstep,1])
                args.append((np.array(vals), np.array(dvals), mcoords, gcoords))

        results = pool.map(runGaussianProcess, args)

        #runGaussianProcess(args[221342])
        #if os.path.isfile("caw%s.pickle" % (switch)):
        #    buff = open("caw%s.pickle" % (switch), "rb")

        #if os.path.isfile("nugget0/caw.pickle"):
        #    buff = open("nugget0/caw.pickle", "rb")
        #    results = cPickle.load(buff)
        #    buff.close()
        #else:
        #    sys.exit(1) # DEBUG
        #    results = pool.map(runGaussianProcess, args)
        #    # JIC!
        #    buff = open("caw%s.pickle" % (switch), "wb")
        #    cPickle.dump(results, buff)
        #    buff.close()
        
        # results is tsteps x ksteps
        ridx = 0
        for tstep in tsteps:
            print "# SAVING T", tstep
            for key in ksteps:
                result = results[ridx]
                for i in range(len(result[0])):
                    mesonet = mesonets.values()[i]
                    mesonet.pdata[key][tstep,:] = result[0][i], result[1][i]

                ridx += 1

        buff = open(datafile, "wb")
        cPickle.dump(mesonets, buff)
        buff.close()
