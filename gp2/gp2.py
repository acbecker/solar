import sys
import os
import numpy as np
import h5py
import multiprocessing
import cPickle
import matplotlib.pyplot as plt
import itertools
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

fKeys = ("apcp_sfc", "dlwrf_sfc", "dswrf_sfc", "pres_msl", "pwat_eatm",
         "spfh_2m", "tcdc_eatm", "tcolc_eatm", "tmax_2m", "tmin_2m",
         "tmp_2m", "tmp_sfc", "ulwrf_sfc", "ulwrf_tatm", "uswrf_sfc")

# Minimal script for gaussian process estimation
class Mesonet(object):
    def __init__(self, stid, nlat, elon, elev, npts):
        self.stid = stid
        self.nlat = nlat
        self.elon = elon
        self.elev = elev

        # Gaussian process interpolation; mean and stdev
        self.pdata = np.recarray((5*11*npts,), dtype={"names": fKeys,
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
        self.data = np.recarray((5*11*npts,), dtype={"names": fKeys,
                                                     "formats": (np.float64, np.float64, np.float64, np.float64, np.float64, 
                                                                 np.float64, np.float64, np.float64, np.float64, np.float64, 
                                                                 np.float64, np.float64, np.float64, np.float64, np.float64)})

def runGaussianProcess((args, regr)):
    nugmin = 0.025**2
    vals, mcoords, gcoords = args

    gp     = GaussianProcess(corr="squared_exponential", 
                             regr=regr,
                             theta0=1e-1, thetaL=1e-2, thetaU=1,
                             normalize=True,
                             nugget=nugmin,
                             random_start=1)

    try:
        gpres = gp.fit(gcoords, vals)
        pred = gp.predict(mcoords) #, eval_MSE=True) # Speed this thing up!
    except:
        import pdb; pdb.set_trace()
    #pred, varpred = gp.predict(mcoords) #, eval_MSE=True) # Speed this thing up!
    #sigpred = np.sqrt(varpred)
    
    return pred, #, sigpred

if __name__ == "__main__":
    switch   = sys.argv[1]
    if switch == "train":
        npts = 5113
    else:
        npts = 1796

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
    for key in fKeys:
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
                data = f[fMapper[key]][:,:,:,latidx,lonidx] # 1796, 11, 5
                gefs.data[key][:] = np.ravel(data)        # data[1][0][0] = 9.0; np.ravel(data)[11*5] = 9.0
                sidx += 1
    
    # Mesonet coords
    mlats = []
    mlons = []
    melevs = []
    for mesonet in mesonets.values(): # MAKE SURE COORDS ARE READ IN THE SAME ORDER AS VALUES
        mlats.append(mesonet.nlat)
        mlons.append(mesonet.elon)
        melevs.append(mesonet.elev)
    mlats   = np.array(mlats)
    mlons   = np.array(mlons)
    melevs  = np.array(melevs)
    #mcoords = np.array(zip(mlats,mlons,melevs))
    mcoords = np.array(zip(mlats,mlons))

    # GEFS coordsgef
    glats = []
    glons = []
    gelevs = []
    for gefs in gefss.values():  # MAKE SURE COORDS ARE READ IN THE SAME ORDER AS VALUES
        glats.append(gefs.nlat) 
        glons.append(gefs.elon)
        gelevs.append(gefs.elev)
    glats   = np.array(glats)
    glons   = np.array(glons)
    gelevs  = np.array(gelevs)
    #gcoords = np.array(zip(glats,glons,gelevs))
    gcoords = np.array(zip(glats,glons))

    # Get ready to run all the GP
    pool = multiprocessing.Pool(multiprocessing.cpu_count()//2) # high mem!

    tsteps = range(npts * 11 * 5)

    #for dologit in (False, True):
    for dologit in (False,):
        if dologit:
            if os.path.isfile("logit.pickle"):
                print "# READING LOGIT PICKLE"
                buff        = open("logit.pickle", "rb")
                fmins,fmaxs = cPickle.load(buff)
                buff.close()
                # Warning that this was set with the test data, hopefully the training data don't have different ranges...?
                for key in fKeys:
                    gefs.data[key] -= fmins[key]
                    gefs.data[key] /= fmaxs[key] * 1.02
                    gefs.data[key] += 0.01
                    logit = np.log(gefs.data[key] / (1.0 - gefs.data[key]))
                    gefs.data[key]  = logit
            else:
                sys.exit(1)
                # logit.py should now do this 

                fmins = {}
                fmaxs = {}
                for key in fKeys:
                    fmins[key]      = gefs.data[key].min()
                    gefs.data[key] -= fmins[key]
                    fmaxs[key]      = gefs.data[key].max()
                    gefs.data[key] /= fmaxs[key] * 1.02
                    gefs.data[key] += 0.01
                    logit = np.log(gefs.data[key] / (1.0 - gefs.data[key]))
                    gefs.data[key]  = logit

        args   = []
        for tstep in tsteps:
            print "# PREPPING T", tstep
            for key in fKeys:
                vals  = []
                for gefs in gefss.values(): # MAKE SURE VALUES ARE READ IN THE SAME ORDER AS COORDS
                    vals.append(gefs.data[key][tstep])
                args.append((np.array(vals), mcoords, gcoords))

        for regr in ("constant", "linear", "quadratic"):
            #results = pool.map(runGaussianProcess, itertools.izip(args, itertools.repeat(regr)))
            results = []
            for arg in args:
                results.append(runGaussianProcess((arg, regr)))
    
            ridx = 0
            for tstep in tsteps:
                print "# SAVING T", tstep
                for key in fKeys:
                    result = results[ridx]
                    for i in range(len(result[0])):
                        mesonet = mesonets.values()[i] # MAKE SURE VALUES ARE READ IN THE SAME ORDER AS COORDS
                        mesonet.pdata[key][tstep] = result[0][i]
                    ridx += 1
        
            if dologit:
                datafile = "gp2b_%s_%s_logit.pickle" % (switch, regr)
                buff = open(datafile, "wb")
                cPickle.dump((mesonets, fmins, fmaxs), buff)
                buff.close()
            else:
                datafile = "gp2b_%s_%s.pickle" % (switch, regr)
                buff = open(datafile, "wb")
                cPickle.dump(mesonets, buff)
                buff.close()
    
            del results
    
