# FRFbathy_interp
"""
This script is based on an original Matlab script by Bonnie Ludka
Rewritten in python by Dylan Anderson on 7/25/2019

It is written for LARC bathy interpolation and produces a number of plots at each step to check the processing
"""
import netCDF4 as nc
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from scipy import interpolate
import datetime as DT
from objMapPrep import coarseBackground, binMorph
from getdatatestbed import getDataFRF
from testbedutils import sblib as sb
import objMapPlots
from objMapInterp import map_interp
from makeNetCDF import py2netCDF as p2nc

noise = 0.1        # meters elevation
Lx = 20             # meters cross-shore
Ly = 150            # meters alongshore
countBinThresh = 3  # number of data points required in bin
nmseThresh = 0.7    # threshold for including data into the final product based on nmse Estimate
# backgroundGridLoc = 'http://134.164.129.55/thredds/dodsC/cmtb/grids/TimeMeanBackgroundDEM/backgroundDEMt0_TimeMean.nc'
putFilesHere = "figures/objMapQAQC"
outFpath = "outNetCDFfiles"
###########
# set start, end
start, end = DT.datetime(2010, 3, 17), DT.datetime(2020, 2, 11)


# Now lets reduce to a new coarser grid that will have multiple observations in each bin
cdx = Lx/2  # 100 # crossshore res ~8m (?)
cdy = Ly/2  # 10  # alongshore res ~50m (?)
plt.ioff()

def findBestBackground(ncfileLoc, date, **kwargs):
    backgroundFname = kwargs.get('background',
                             "http://134.164.129.55/thredds/dodsC/cmtb/grids/TimeMeanBackgroundDEM/backgroundDEMt0_TimeMean.nc")

    globList = sorted(glob.glob(os.path.join(ncfileLoc, '*.nc')))
    listDateDiffs =np.array([DT.datetime.strptime(f.split('_')[-1], '%Y%m%d.nc') for f in globList]) - date
    yesterdayFile = globList[np.argmax(listDateDiffs[listDateDiffs<DT.timedelta(0)]).squeeze()]
    if os.path.isfile(yesterdayFile):
        # load yesterdays file for background
        print('    background: {}'.format(yesterdayFile))
        ncfile = nc.Dataset(yesterdayFile)
        xFRFbackground = ncfile['xFRF'][:]
        yFRFbackground = ncfile['yFRF'][:]
        zbg = ncfile['elevation'][:].squeeze()
    else:
        background = nc.Dataset(backgroundFname)
        xFRFbackground = background['xFRF'][:]
        yFRFbackground = background['yFRF'][:]
        zbg = background['elevation'][:]
    
    return xFRFbackground, yFRFbackground, zbg
def buildCollectiveXYZ(bathy, dune, pier, claris):
    zs, ys, xs, types = [], [], [], []
    if bathyAll is not None and bathy is not None:
        # this is not a grid so does not need to be mesh gridded like others
        zs = np.ma.concatenate([zs, bathy['elevation'].flatten()])
        ys = np.ma.concatenate([ys, bathy['yFRF'].flatten()])
        xs = np.ma.concatenate([xs, bathy['xFRF'].flatten()])
        types.append(' bathy')
    if duneAll is not None and dune is not None:
        xx, yy = np.meshgrid(dune['xFRF'], dune['yFRF'])
        # repeat in time xxFRFbackground and yyFRFbackground grid
        allXs = np.tile(xx, [np.size(dune['time']), 1 , 1])
        allYs = np.tile(yy, [np.size(dune['time']), 1 , 1])
        # now eliminate xxFRFbackground/yyFRFbackground values that the dune['elevation'] values are masked (True)
        xs = np.ma.concatenate([xs, allXs[~dune['elevation'].mask].flatten()])
        ys = np.ma.concatenate([ys, allYs[~dune['elevation'].mask].flatten()])
        zs = np.ma.concatenate([zs, dune['elevation'][~dune['elevation'].mask].flatten()])
        types.append(' dune')
    if pierAll is not None and pier is not None:
        xx, yy = np.meshgrid(pier['xFRF'], pier['yFRF'])
        # repeat in time xxFRFbackground and yyFRFbackground grid
        allXs = np.tile(xx, [np.size(pier['time']), 1 , 1])
        allYs = np.tile(yy, [np.size(pier['time']), 1 , 1])
        # now eliminate xxFRFbackground/yyFRFbackground values that the dune['elevation'] values are masked (True)
        xs = np.ma.concatenate([xs, allXs[~pier['elevation'].mask].flatten()])
        ys = np.ma.concatenate([ys, allYs[~pier['elevation'].mask].flatten()])
        zs = np.ma.concatenate([zs, pier['elevation'][~pier['elevation'].mask].flatten()])
        types.append(' pier')
    if clarisAll is not None and claris is not None:
        xx, yy = np.meshgrid(claris['xFRF'], claris['yFRF'])
        # repeat in time xxFRFbackground and yyFRFbackground grid
        allXs = np.tile(xx, [np.size(claris['time']), 1 , 1])
        allYs = np.tile(yy, [np.size(claris['time']), 1 , 1])
        # now eliminate xxFRFbackground/yyFRFbackground values that the dune['elevation'] values are masked (True)
        xs = np.ma.concatenate([xs, allXs[~claris['elevation'].mask].flatten()])
        ys = np.ma.concatenate([ys, allYs[~claris['elevation'].mask].flatten()])
        zs = np.ma.concatenate([zs, claris['elevation'][~claris['elevation'].mask].flatten()])
        types.append(' claris')
    
    return xs, ys, zs, types
dt = DT.timedelta(days=1)

go = getDataFRF.getObs(start, end)
bathyAll = go.getBathyTransectFromNC(forceReturnAll=True)

for tt, date in enumerate(sb.createDateList(start, end, dt)):   # make one merged product (hourly)
    print('--- making grid for {}'.format(date))
    ###################
    # now gather data #
    ###################
    # Gather Data for today's interpolations
    go = getDataFRF.getObs(date, date+dt)
    # now gather data
    pierAll = go.getLidarDEM(lidarLoc='pier')
    duneAll = go.getLidarDEM(lidarLoc='dune')
    clarisAll = None # go.getLidarDEM(lidarLoc='claris')
    
    # 1. Pick appropriate background file
    xFRFbackground, yFRFbackground, zbg = findBestBackground(outFpath, date)
    
    xxFRFbackground, yyFRFbackground = np.meshgrid(xFRFbackground, yFRFbackground)
    xxFRFbackground = np.ma.masked_where(yyFRFbackground > 4500, xxFRFbackground)
    yyFRFbackground = np.ma.masked_where(yyFRFbackground > 4500, yyFRFbackground)
    zb = np.ma.masked_where(yyFRFbackground > 4500, zbg)
    xCoarse, yCoarse, zCoarse, xn, yn = coarseBackground(x=xFRFbackground, y=yFRFbackground, z=zbg, cdx=cdx, cdy=cdy)

    # what file string should i label plots with?
    fileString = "{}_{}_{}_{}".format(date.strftime('%Y%m%d'), Lx, Ly, nmseThresh)
    
    # isolate data which data to include from larger data call above
    if bathyAll is not None:
        idxBathy = np.argwhere((bathyAll['time'] < date + dt) & (bathyAll['time'] > date)).squeeze()
        # need to remove xFRF, yFRF from default exempt list because x,y points are co-located to elvations
        bathy = sb.reduceDict(bathyAll, idxBathy, exemptList=['time', 'name', 'wavefreqbin'])
    if duneAll is not None:
        idxDune = np.argwhere((duneAll['time'] < date + dt) & (duneAll['time'] > date)).squeeze()
        dune = sb.reduceDict(duneAll, idxDune)
    else:
        dune = None
    if pierAll is not None:
        idxPier = np.argwhere((pierAll['time'] < date + dt) & (pierAll['time'] > date)).squeeze()
        pier = sb.reduceDict(pierAll, idxPier)
    else:
        pier = None
        
    if clarisAll is not None:
        idxClaris = np.argwhere((clarisAll['time'] < date + dt) & (clarisAll['time'] > date)).squeeze()
        claris = sb.reduceDict(clarisAll, idxClaris)
    else:
        claris=None
    if bathyAll is None and dune is None and claris is None and pier is None:
        continue
    
    # concatenate available data to bin and grid
    
    xs,ys,zs,types = buildCollectiveXYZ(bathy, dune, pier, claris)
    if len(types) == 0:
        print('    No available Data to integrate')
        continue
    print("incorporating {} data".format(types))
    # bin the data (xs, ys, zs) into xn and yn boundaries
    binned = binMorph(xn, yn, xs, ys, zs)
    
    bc = binned['binCounts']
    cellIDsAboveBinThresh = np.nonzero(bc > countBinThresh)
    if np.size(cellIDsAboveBinThresh) == 0:
        continue
    bcvals = bc[cellIDsAboveBinThresh]
    zbin = binned['zBinVar']
    # check standard error to see if noise value you chose is reasonable
    stdErr = np.sqrt(zbin[cellIDsAboveBinThresh] / bc[cellIDsAboveBinThresh])
    zFluc = binned['zBinMedian'][cellIDsAboveBinThresh] - zCoarse[cellIDsAboveBinThresh]
    
    ofname = os.path.join(putFilesHere, fileString +'_preprocess.png')
    objMapPlots.binnedDataPlot(ofname, binned, xCoarse, yCoarse, cellIDsAboveBinThresh, stdErr, zFluc)

    # extract relevant domain from background Values for mapping from coarser grid scale add some for the smoothing
    #   eg Lx/y * 3
    xmin = np.min(xCoarse[cellIDsAboveBinThresh]) - Lx * 3                                       # min cross-shore
    xmax = np.max(xCoarse[cellIDsAboveBinThresh]) + Lx * 3                                       # max cross-shore
    ymin = np.min(yCoarse[cellIDsAboveBinThresh]) - Ly * 3                                       # min alongshore
    ymax = np.max(yCoarse[cellIDsAboveBinThresh]) + Ly * 3                                       # max alongshore
    # find appropriate indices that are in the subsection of the domain
    idInt = np.where((xCoarse > xmin) & (xCoarse < xmax) & (yCoarse > ymin) & (yCoarse < ymax))
    
    # checking the index created
    ofname = os.path.join(putFilesHere, fileString + '_indexCheck.png')
    objMapPlots.scatterDEM(x=xCoarse[idInt], y=yCoarse[idInt], z=zCoarse[idInt], title='Background Grid Values '
                           'defined by subset', label='elevation [m]', ofname=ofname)
    
    dgcov, dcovE, A, Aprime, mapFluc, nmseEst, dcovA, dcovA2, sigVar = map_interp(x=xCoarse[cellIDsAboveBinThresh],
                                                                                  y=yCoarse[cellIDsAboveBinThresh],
                                                                                  zFluc=zFluc,
                                                                                  noise=noise, Lx=Lx, Ly=Ly,
                                                                                  xInt=xCoarse[idInt],
                                                                                  yInt=yCoarse[idInt])

    # initialize arrays
    allzeros = np.zeros_like(xCoarse)
    nmseest = np.ones_like(xCoarse)
    mapfluc = np.zeros_like(xCoarse)
    
    nmseest[idInt] = nmseEst.T
    mapfluc[idInt] = mapFluc
    goodi = np.nonzero(nmseest <= nmseThresh)
    badi = np.nonzero(nmseest > nmseThresh)
    mapfluc[badi] = allzeros[badi]
    
    mapz = mapfluc + zCoarse
    
    # look at error estimates
    if len(goodi[0]) == 0:  # there's no good points
        continue
    ofname=os.path.join(putFilesHere, fileString +'_postprocessing.png')
    objMapPlots.postProcessedGridPlot(ofname, xCoarse, yCoarse, nmseest, mapfluc, mapz, goodi)
    
    # now lets interpolate these accepted values to the 5 x 5 m grid
    xgood = xCoarse.flatten()
    ygood = yCoarse.flatten()
    points = np.vstack([xgood, ygood])
    points = points.T
    zgood = mapz.flatten()
    # interpolate coarse grid to the background values
    fi = interpolate.griddata(points=points, values=zgood, xi=(xxFRFbackground, yyFRFbackground), method='linear')
    
    #zcfi = fi(xc, yc)
    # objMapPlots.scatterDEM(xxFRFbackground, yyFRFbackground, fi, label='elevation[m]', cmap='ocean_r',
    #                        title='Merging products where Error acceptable')
    
    print('add capability to see how many points have been included in new grid')
    #################
    # do comparison of residuals at each survey point subtracted from grid as scatter plotted over grid and 1-1 plot
    # do cross-shore profile and alongshore profile at a few points
    outFname = os.path.join(putFilesHere, "QAQCplot_"+ fileString +".png")
    diff = fi - zbg
    pierRMSE, duneRMSE, bathyRMSE = objMapPlots.QAQCplot(outFname, xxFRFbackground, yyFRFbackground, fi,  bathy,
                                                dune, pier, diff=diff, xbounds=[0, 1200], ybounds=[-200, 1500])
    ########## preprocess for netCDF output as needed #############
    fi = np.expand_dims(fi.squeeze(), 0)
    updateTime = (np.ones_like(fi) * -999)
    mask = diff != 0
    updateTime[0][mask] = date.timestamp()
    
    out = {'xFRF':       xFRFbackground,
           'yFRF':       yFRFbackground,
           'elevation':  fi,
           'time':       np.expand_dims(date.timestamp(), 0),
           'latitude':   np.ones_like(fi) * -999,
           'longitude':  np.ones_like(fi) * -999,
           'y_smooth':   np.ones_like(np.expand_dims(date.timestamp(), 0)) * -999,
           'updateTime': updateTime,
           'RMSEvals': np.expand_dims([pierRMSE, duneRMSE, bathyRMSE],0),
           'RMSEval': [1,2,3]}
    
    ofname = os.path.join(outFpath, 'CMTB_integratedBathy_fused_{}.nc'.format(date.strftime("%Y%m%d")))
    varYml = "/home/spike/repos/makebathyinterp/yamls/IntegratedBathy_grid_var.yml"
    globalYml = "/home/spike/repos/makebathyinterp/yamls/BATHY/FRFti_global.yml"
    p2nc.makenc_generic(ofname, globalYml, varYml, out)
