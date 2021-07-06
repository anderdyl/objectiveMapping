"""This script is designed to generate the Average Bathy/topo to be used for objective mapping or other methods """
import netCDF4 as nc
from scipy import interpolate
import numpy as np
from osgeo import gdal, osr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pickle, sys
sys.path.append('/home/spike/repos/makebathyinterp')
from testbedutils import geoprocess as gp
######################################
# define boundaries for domain of interest
ybounds = [-200,2000]
xbounds = [0, 1300]
dx = 5
outFname = 'backgroundPickle.pickle'
# where is my source data
ncfile = nc.Dataset('http://134.164.129.55/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/surveyTransects.ncml')
surveyNums = np.unique(ncfile['surveyNumber'][:])
# What does the output look like?
xFRFout = np.arange(xbounds[0], xbounds[1]+dx, dx)
yFRFout = np.arange(ybounds[0], ybounds[1]+dx, dx)
allGrids = np.ones([len(surveyNums), len(xFRFout), len(yFRFout)]) * -999

for tt, sn in enumerate(surveyNums):
    # identify as survey number
    idxSN = ncfile['surveyNumber'][:] == sn
    elevations = ncfile['elevation'][idxSN]
    xFRF = ncfile['xFRF'][idxSN]
    yFRF = ncfile['yFRF'][idxSN]
    profileNumbers = np.unique(ncfile['profileNumber'][idxSN])
    print('survey Number {}: {} of {}'.format(sn, tt, len(surveyNums)))
    # check to see if there are pier lines
    if (np.abs(profileNumbers - 515) < 10).sum() < 2:
        continue  # go to next survey
    
    ## remove data points from xbounds
    elevations = elevations[xFRF<=max(xbounds)]
    yFRF = yFRF[xFRF<=max(xbounds)]
    xFRF = xFRF[xFRF<=max(xbounds)]
    
    elevations = elevations[xFRF>=min(xbounds)]
    yFRF = yFRF[xFRF>=min(xbounds)]
    xFRF = xFRF[xFRF>=min(xbounds)]
    
    ## remove data points from ybounds
    elevations = elevations[yFRF<=max(ybounds)]
    xFRF = xFRF[yFRF<=max(ybounds)]
    yFRF = yFRF[yFRF<=max(ybounds)]
    
    elevations = elevations[yFRF>=min(ybounds)]
    xFRF = xFRF[yFRF>=min(ybounds)]
    yFRF = yFRF[yFRF>=min(ybounds)]
    
    
    # grid each survey
    try:
        f = interpolate.interp2d(xFRF, yFRF, elevations, bounds_error=False, fill_value=-999)
        allGrids[tt] = f(xFRFout, yFRFout)
    except:
        continue
    with open(outFname, 'wb') as fid:
        pickle.dump(allGrids, protocol=pickle.HIGHEST_PROTOCOL)
# fill with mask values if no pier lines
# average in time.
############################
url = "http://134.164.129.55/thredds/dodsC/cmtb/grids/TimeMeanBackgroundDEM/backgroundDEMt0_TimeMean.nc"
ncfile = nc.Dataset(url)
idxX = (ncfile['xFRF'][:]>xbounds[0] ) & (ncfile['xFRF'][:] < xbounds[1])
idxY = (ncfile['yFRF'][:]>ybounds[0] ) & (ncfile['yFRF'][:] < ybounds[1])
meanElevation = ncfile['elevation'][idxY, idxX]
meanXfrf = ncfile['xFRF'][idxX]
meanYfrf = ncfile['yFRF'][idxY]
########### load jbltx data for background
fname = "/home/spike/repos/pyObjectiveMapping/Job556221_nc2019_dunex.tif"
f = gdal.Open(fname)
one = f.GetRasterBand(1).ReadAsArray()
one = np.ma.array(one, mask=one==-999999)
upperLeftX, xRes, _, upperLeftY, _, yRes = f.GetGeoTransform()
lons = np.arange(upperLeftX, xRes*one.shape[1]+upperLeftX, xRes)
lats = np.arange(upperLeftY, yRes*one.shape[0]+upperLeftY, yRes)
# create unique points
xxLons, yyLats = np.meshgrid(lons, lats)
xFRF, yFRF = [], []
for coord in zip(xxLons.flatten(), yyLats.flatten()):
    coordOut = gp.FRFcoord(coord[0],coord[1])
    xFRF.append(coordOut['xFRF'])
    yFRF.append(coordOut['yFRF'])
xxFRF, yyFRF = np.meshgrid(xFRF, yFRF)
########


# improj      =       f.GetProjection()
# inproj_B05      =       osr.SpatialReference()
# inproj_B05.ImportFromWkt(improj)
# projcs_B05      =       inproj_B05.GetAuthorityCode('PROJCS')
# projection_B05  =       ccrs.epsg(projcs_B05)
