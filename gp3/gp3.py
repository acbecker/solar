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

notes = """
In this case, we do a 3-d gaussian interpolation.  The elevations are read from

  h5py.File("gefs_elevations.nc")["elevation_control"]

This is the altitude of the first of the 11 ensembles.  The other 10
runs appear to be evaluated at

  h5py.File("gefs_elevations.nc")["elevation_perturbation"]

In this case, I will read in the 10 elevations, sigma clip them, and
there will be 2 altitude inputs to the 3-d GP.
"""

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

        # Gaussian process interpolation; mean 
        self.pdata = np.recarray((npts,5), dtype={"names": fKeys,
                                                  "formats": (np.float64, np.float64, np.float64, np.float64, np.float64, 
                                                              np.float64, np.float64, np.float64, np.float64, np.float64, 
                                                              np.float64, np.float64, np.float64, np.float64, np.float64)})
        
class GEFS(object):
    def __init__(self, stid, nlat, elon, elev, npts):
        self.stid = stid
        self.nlat = nlat
        self.elon = elon
        self.elev = elev

        # Capture mean
        self.data = np.recarray((npts,5), dtype={"names": fKeys,
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

    gpres = gp.fit(gcoords, vals)
    pred = gp.predict(mcoords) 
    return pred

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
    
    gefssC = {}
    gefssE = {}
    for key in fKeys:
        print "# LOADING", key
    
        if switch == "train":
            f = h5py.File("../train/%s_latlon_subset_19940101_20071231.nc" % (key), "r")
        else:
            f = h5py.File("../test/%s_latlon_subset_20080101_20121130.nc" % (key), "r")
        
        if len(gefssC.keys()) == 0:
            print "# INITIALIZING GEFS"
            sidx = 0
            for latidx in range(len(f['lat'])):
                for lonidx in range(len(f['lon'])):
                    gefssC[sidx] = GEFS(sidx, f["lat"][latidx], f["lon"][lonidx]-360., 0.0, npts)
                    gefssE[sidx] = GEFS(sidx, f["lat"][latidx], f["lon"][lonidx]-360., 0.0, npts)
                    sidx += 1

            f2 = h5py.File("../gefs_elevations.nc")
            sidx = 0
            for latidx in range(9):
                for lonidx in range(16):
                    gefssC[sidx] = GEFS(sidx, 
                                        f2["latitude"][latidx][lonidx], 
                                        f2["longitude"][latidx][lonidx]-360., 
                                        f2["elevation_control"][latidx][lonidx], npts)
                    gefssE[sidx] = GEFS(sidx, 
                                        f2["latitude"][latidx][lonidx], 
                                        f2["longitude"][latidx][lonidx]-360., 
                                        f2["elevation_perturbation"][latidx][lonidx], npts)
                    sidx += 1
            
    
        sidx = 0
        for latidx in range(9):
            for lonidx in range(16):
                dataC = f[fMapper[key]][:,0,:,latidx,lonidx] 
                dataE = sigclip(f[fMapper[key]][:,1:,:,latidx,lonidx], True) 

                gefsC = gefssC[sidx]
                gefsE = gefssE[sidx]

                # make sure indices make sense
                assert( gefsC.nlat == f["lat"][latidx] )
                assert( gefsC.elon == f["lon"][lonidx]-360.0 )
                gefsC.data[key] = dataC
                gefsE.data[key] = dataE

                sidx += 1
    
    # Mesonet coords
    mlats  = []
    mlons  = []
    melevs = []
    for mesonet in mesonets.values(): # MAKE SURE COORDS ARE READ IN THE SAME ORDER AS VALUES
        mlats.append(mesonet.nlat)
        mlons.append(mesonet.elon)
        melevs.append(mesonet.elev)
    mlats   = np.array(mlats)
    mlons   = np.array(mlons)
    melevs  = np.array(melevs)
    mcoords = np.array(zip(mlats,mlons,melevs))

    # GEFS coords
    glats = []
    glons = []
    gelevs = []
    for gefs in gefssC.values():  # MAKE SURE COORDS ARE READ IN THE SAME ORDER AS VALUES
        glats.append(gefs.nlat) 
        glons.append(gefs.elon)
        gelevs.append(gefs.elev)
    for gefs in gefssE.values():  # MAKE SURE COORDS ARE READ IN THE SAME ORDER AS VALUES
        glats.append(gefs.nlat) 
        glons.append(gefs.elon)
        gelevs.append(gefs.elev)
    glats   = np.array(glats)
    glons   = np.array(glons)
    gelevs  = np.array(gelevs)
    gcoords = np.array(zip(glats,glons,gelevs))

    # Get ready to run all the GP
    pool = multiprocessing.Pool(multiprocessing.cpu_count()//2) # high mem!

    args   = []
    for tstep in range(npts):
        print "# PREPPING T", tstep
        for fstep in range(5):
            for key in fKeys:
                vals  = []
                for gefs in gefssC.values(): # MAKE SURE VALUES ARE READ IN THE SAME ORDER AS COORDS
                    vals.append(gefs.data[key][tstep][fstep])
                for gefs in gefssE.values(): # MAKE SURE VALUES ARE READ IN THE SAME ORDER AS COORDS
                    vals.append(gefs.data[key][tstep][fstep])
                args.append((np.array(vals), mcoords, gcoords))

    # vals: 288, 
    # mcoord: 98, 3
    # gcoord: 288, 3

    for regr in ("constant", "linear", "quadratic"):
        results = pool.map(runGaussianProcess, itertools.izip(args, itertools.repeat(regr)))
    
        ridx = 0
        for tstep in range(npts):
            print "# SAVING T", tstep
            for fstep in range(5):
                for key in fKeys:
                    result = results[ridx]
                    for i in range(len(result)):
                        mesonets.values()[i].pdata[key][tstep][fstep] = result[i]
                    ridx += 1
    
        datafile = "gp3_%s_%s.pickle" % (switch, regr)
        buff = open(datafile, "wb")
        cPickle.dump(mesonets, buff)
        buff.close()
    
        del results
    
