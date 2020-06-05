# FRFbathy_interp
"""
This script is based on an original Matlab script by Bonnie Ludka
Rewritten in python by Dylan Anderson on 7/25/2019

It is written for LARC bathy interpolation and produces a number of plots at each step to check the processing
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import datetime as DT
import h5py
from objMapPrep import coarseBackground
from objMapPrep import getGeoDatasets
from objMapPrep import binMorph
from objMapPlots import scatterDEM, pcolorDEM
from objMapInterp import map_interp
from getdatatestbed import getDataFRF
from testbedutils import sblib as sb
import objMapPlots

noise = 0.01    # meters elevation
Lx = 15    # meters cross-shore
Ly = 150    # meters alongshore
countBinThresh = 3  # number of data points required in bin

backgroundGridLoc = 'http://134.164.129.55/thredds/dodsC/cmtb/grids/TimeMeanBackgroundDEM/backgroundDEMt0_TimeMean.nc'
###########
# set start, end
start, end = DT.datetime(2020,1,2), DT.datetime(2020,1,3)

###################
# now gather data #
###################
# Gather Data for today's interpolations
go = getDataFRF.getObs(start, end)
gtb = getDataFRF.getDataTestBed(start, end)

pierAll = go.getLidarDEM(lidarLoc='pier')
duneAll = go.getLidarDEM(lidarLoc='dune')
bathyAll = go.getBathyTransectFromNC()
clarisAll = go.getLidarDEM(lidarLoc='claris')
background = gtb.getBathyIntegratedTransect()
######################### can i put below in function #######################
# Load the background mean
# bg = Dataset(backgroundGridLoc)
x = background['xFRF']
y = background['yFRF'][:]
zbg = background['elevation']
xx, yy = np.meshgrid(x, y)

xx = np.ma.masked_where(yy > 4500, xx)
yy = np.ma.masked_where(yy > 4500, yy)
zb = np.ma.masked_where(yy > 4500, zbg)
# Now lets reduce to a new coarser grid that will have multiple observations in each bin
# alongshore res ~50m (?)
# crossshore res ~8m (?)
cdx = 50
cdy = 8
xc, yc, zc, xn, yn = coarseBackground(x=x, y=y, z=zbg, cdx=cdx, cdy=cdy)
######################### can i put above in function #######################

dt = DT.timedelta(days=60)
for tt, date in enumerate(sb.createDateList(start, end, dt)):   # make one merged product (hourly)
    if bathyAll is not None:
        print('   add logic to bathy, this is going to be weird')
        bathy = sb.reduceDict(bathyAll, np.argwhere(bathyAll['time'] == date).squeeze())
    if duneAll is not None:
        dune = sb.reduceDict(duneAll, np.argwhere(duneAll['time'] == date).squeeze())
    if clarisAll is not None:
        claris = sb.reduceDict(clarisAll, np.argwhere(clarisAll['time'] == date).squeeze())
    if pierAll is not None:
        pier = sb.reduceDict(pierAll, np.argwhere(pierAll['time'] == date).squeeze())
    if bathyAll is None and dune is None and claris is None and pier is None:
        continue
    
    # concatenate available data to bin and grid
    zs, ys, xs, types = [], [], [], []
    if bathyAll is not None and bathy is not None:
        xx, yy = np.meshgrid(bathy['xFRF'], bathy['yFRF'])
        zs = np.ma.concatenate([zs, bathy['elevation'].flatten()])
        ys = np.ma.concatenate([ys, yy.flatten()])
        xs = np.ma.concatenate([xs, xx.flatten()])
        types.append(' bathy')
    if duneAll is not None and dune is not None:
        xx, yy = np.meshgrid(dune['xFRF'], dune['yFRF'])
        zs = np.ma.concatenate([zs, dune['elevation'].flatten()])
        ys = np.ma.concatenate([ys, yy.flatten()])
        xs = np.ma.concatenate([xs, xx.flatten()])
        types.append(' dune')
    if pierAll is not None and pier is not None:
        xx, yy = np.meshgrid(pier['xFRF'], pier['yFRF'])
        zs = np.ma.concatenate([zs, pier['elevation'].flatten()])
        ys = np.ma.concatenate([ys, yy.flatten()])
        xs = np.ma.concatenate([xs, xx.flatten()])
        types.append(' pier')
    if clarisAll is not None and claris is not None:
        xx, yy = np.meshgrid(claris['xFRF'], claris['yFRF'])
        zs = np.ma.concatenate([zs, claris['elevation'].flatten()])
        ys = np.ma.concatenate([ys, yy.flatten()])
        xs = np.ma.concatenate([xs, xx.flatten()])
        types.append(' claris')
   
    print("incorporating {} data".format(types))
    binned = binMorph(xn, yn, xs, ys, zs)
    
    bc = binned['binCounts']
    cellIDsAboveBinThresh = np.nonzero(bc > countBinThresh)
    bcvals = bc[cellIDsAboveBinThresh]
    zbin = binned['zBinVar']
    # check standard error to see if noise value you chose is reasonable
    stdErr = np.sqrt(zbin[cellIDsAboveBinThresh] / bc[cellIDsAboveBinThresh])
    zbM = binned['zBinMedian']
    zFluc = zbM[cellIDsAboveBinThresh] - zc[cellIDsAboveBinThresh]
    
    ofname = 'preprocess.png'
    objMapPlots.binnedDataPlot(ofname, binned, xc, yc, cellIDsAboveBinThresh, stdErr, zFluc)
    # scatterDEM(x=binned['xBinMedian'], y=binned['yBinMedian'], z=binned['zBinMedian'],
    #            title="Binned Survey", label='elevation (m)')
    #
    # pcolorDEM(x=xc, y=yc, z=binned['binCounts'], title="Binned observations", label='data points per bin')
    #
    # scatterDEM(x=xc[id], y=yc[id], z=stdErr, title='standard error [m]', label='standard error [m]')
    #
    # scatterDEM(x=xc[id], y=yc[id], z=zFluc, title='Difference between Median Obs and previous map',
    #            label='elevation flucuation [m] of binned data', cmap='RdBu')
    #
    
    
    # map example survey
    # extract relevant domain for mapping from coarser grid scale
    xmin = np.min(xc[cellIDsAboveBinThresh]) - Lx * 3                                               # min cross-shore
    xmax = np.max(xc[cellIDsAboveBinThresh]) + Lx * 3                                               # max cross-shore
    ymin = np.min(yc[cellIDsAboveBinThresh]) - Ly * 3                                               # min alongshore
    ymax = np.max(yc[cellIDsAboveBinThresh]) + Ly * 3                                               # max alongshore
    idInt = np.where((xc > xmin) & (xc < xmax) & (yc > ymin) & (yc < ymax))  # find appropriate indices that meet
    
    # checking the index created
    ofname = 'indexCheck.png'
    scatterDEM(x=xc[idInt], y=yc[idInt], z=zc[idInt], title='subset defined by index', label='elevation [m]',
               ofname=ofname)
    
    dgcov, dcovE, A, Aprime, mapFluc, nmseEst, dcovA, dcovA2, sigVar = map_interp(x=xc[cellIDsAboveBinThresh], y=yc[cellIDsAboveBinThresh], zFluc=zFluc,
                                                                                  noise=noise, Lx=Lx, Ly=Ly,
                                                                                  xInt=xc[idInt], yInt=yc[idInt])
    #dgcov, dcovE, A, Aprime, mapFluc, nmseEst, dcovA, dcovA2, sigVar = map_interp(x=xc[id], y=yc[id], zFluc=zFluc,
    #                                                   noise=noise, Lx=Lx, Ly=Ly, xInt=xx[idInt], yInt=yy[idInt])
    # initialize arrays
    allzeros = np.ones(xc.shape)
    nmseest = np.ones(xc.shape)
    mapfluc = np.zeros(xc.shape)
    
    nmseest[idInt] = nmseEst.T
    mapfluc[idInt] = mapFluc
    goodi = np.nonzero(nmseest < .2)
    badi = np.nonzero(nmseest > .2)
    mapfluc[badi] = allzeros[badi]
    
    mapz = mapfluc + zc
    
    # look at error estimates
    fig7, ax7 = plt.subplots(1, 1, figsize=(5, 5))
    sc7 = ax7.pcolormesh(xc, yc, nmseest)
    cbar = plt.colorbar(sc7, ax=ax7)
    #cbar.set_label('elevation [m')
    ax7.set_title('NMSE estimate from map of example survey')
    
    # look at error estimates
    fig8, ax8 = plt.subplots(1, 1, figsize=(5, 5))
    sc8 = ax8.pcolor(xc, yc, mapfluc)
    cbar = plt.colorbar(sc8, ax=ax8)
    #cbar.set_label('elevation [m')
    ax8.set_title('mapped elevation fluctuation example survey')
    
    # look at error estimates
    fig9, ax9 = plt.subplots(1, 1, figsize=(5, 5))
    sc8 = ax9.scatter(xc[goodi], yc[goodi], c=mapfluc[goodi])
    cbar = plt.colorbar(sc8, ax=ax9)
    #cbar.set_label('elevation [m]')
    ax9.set_title('mapped elevation fluctuation (only good values))')
    
    # look at the final map produced
    fig10, ax10 = plt.subplots(1, 1, figsize=(5, 5))
    sc10 = ax10.pcolor(xc, yc, mapz)
    cbar = plt.colorbar(sc10, ax=ax10)
    #cbar.set_label('elevation [m]')
    ax10.set_title('final mapped elevation example survey [m]')

    # now lets interpolate these accepted values to the 5 x 5 m grid
   
    xgood = xc.flatten()
    ygood = yc.flatten()
    points = np.vstack([xgood, ygood])
    points = points.T
    zgood = mapz.flatten()
    
    fi = interpolate.griddata(points=points, values=zgood, xi=(xx, yy), method='linear')
    
    #zcfi = fi(xc, yc)
    fig10, ax11 = plt.subplots(1, 1, figsize=(5, 5))
    sc2 = ax11.pcolormesh(xx, yy, fi, vmin=-12, vmax=10)
    cbar2 = plt.colorbar(sc2, ax=ax11)
    cbar2.set_label("elevation (m)")
    ax11.set_title("Merging products where error is acceptable: March 3rd, 2018")
    ax11.set_xlim([0, 1300])
    ax11.set_ylim([-150, 1200])
    
    
    
    fig13, ax13 = plt.subplots(1, 1, figsize=(5, 5))
    
    f2 = interpolate.interp2d(x, y, fi, kind='linear')
    zc2 = f2(xn, yn)
    sc2 = ax13.pcolormesh(xc, yc, zc2)
    cbar2 = plt.colorbar(sc2, ax=ax13)
    cbar2.set_label("elevation (m)")
    ax13.set_title("Lxinear interpolation to coarser grid")






bathyFile = 'FRF_20180315_1148_FRF_NAVD88_LARC_GPS_UTC_v20180319.nc'

getData = getGeoDatasets(bathyFile=bathyFile, pierFile=None, duneFile=None, clarisFile=None)

bathy2 = getData.getBathy()
zs2 = bathy2['z']
ys2 = bathy2['y']
xs2 = bathy2['x']


binned2 = binMorph(xn, yn, xs2, ys2, zs2)
zbM2 = binned2['zBinMedian']
bc2 = binned2['binCounts']
id2 = np.nonzero(bc2 > countBinThresh)

zFluc2 = zbM2[id2] - zc2[id2]



fig12, ax12 = plt.subplots(1, 1, figsize=(5, 5))
#sc10 = ax12.scatter(xs2, ys2, c=zs2)
#sc10 = ax12.scatter(binned2['xBinMedian'], binned2['yBinMedian'], c=binned2['zBinMedian'])
#sc10 = ax12.scatter(xc[id2], yc[id2], c=zbM2[id2])
#sc10 = ax12.scatter(xc[id2], yc[id2], c=zc2[id2])
#sc10 = ax12.scatter(binned2['xBinMedian'], binned2['yBinMedian'], c=zFluc2)
sc10 = ax12.scatter(binned2['xBinMedian'], binned2['yBinMedian'], c=binned2['binCounts'])


#sc10 = ax12.scatter(xc[id2], yc[id2], c=zFluc2)

cbar = plt.colorbar(sc10, ax=ax12)
#cbar.set_label('elevation [m')
ax12.set_title('final mapped elevation example survey [m]')





dgcov, dcovE, A, Aprime, mapFluc2, nmseEst, dcovA, dcovA2, sigVar = map_interp(x=xc[id2], y=yc[id2], zFluc=zFluc2, noise=noise, Lx=Lx, Ly=Ly, xInt=xc[idInt], yInt=yc[idInt])



allzeros = np.ones(xc.shape)
nmseest2 = np.ones(xc.shape)
nmseest2[idInt] = nmseEst.T
mapfluc2 = np.zeros(xc.shape)
mapfluc2[idInt] = mapFluc2
goodi2 = np.nonzero(nmseest2 < .2)
badi2 = np.nonzero(nmseest2 > .2)
mapfluc2[badi2] = allzeros[badi2]

mapz2 = mapfluc2+zc2


# look at error estimates
fig9, ax9 = plt.subplots(1, 1, figsize=(5, 5))
sc8 = ax9.pcolor(xc, yc, mapfluc2, vmin=-3, vmax=3, cmap='RdBu')
cbar = plt.colorbar(sc8, ax=ax9)
#cbar.set_label('elevation [m')
ax9.set_title('Updating with Offshore Sandbar Migration 3 weeks later')


xgood = xc.flatten()
ygood = yc.flatten()
points = np.vstack([xgood, ygood])
points = points.T
zgood = mapz2.flatten()
#zgood = mapfluc2.flatten()

fi2 = interpolate.griddata(points=points, values=zgood, xi=(xx, yy), method='linear')
f3 = interpolate.interp2d(x, y, fi2, kind='linear')
zc3 = f3(xn, yn)

fig13, ax13 = plt.subplots(1, 1, figsize=(5, 5))
sc2 = ax13.pcolor(xc, yc, zc3)
cbar2 = plt.colorbar(sc2, ax=ax13)
cbar2.set_label("elevation (m)")
ax13.set_title("Linear interpolation to coarser grid")


# compare to survey
# get a subset of the yc
transect = np.where((yc > 1050) & (yc < 1150))
fig12, ax12 = plt.subplots(1, 1, figsize=(5, 5))
sc10 = ax12.scatter(xc[transect], zc3[transect])

realtransect = np.where((ys2 > 1050) & (ys2 < 1150))
sc10b = ax12.scatter(xs2[realtransect], zs2[realtransect], c='k')







#zs = np.ma.concatenate([bathy['z'], dune['z'], pier['z'], claris['z']])
#ys = np.ma.concatenate([bathy['y'], dune['y'], pier['y'], claris['y']])
#xs = np.ma.concatenate([bathy['x'], dune['x'], pier['x'], claris['x']])






