



from netCDF4 import Dataset
import numpy as np
import h5py
from testbedutils import geoprocess
from matplotlib import path


def binMorph(xn, yn, xs, ys, zs):
    """We continuously need to take a spatial dataset and bin it, or rasterize it,
    to a gridded product of lower resolution

    Args:
        xn (array): bin-corners in the x-direction
        yn (array): bin-corners in the y-direction
        xs (array): x-coordinates for spatial data being binned
        ys (array): y-coordinates for spatial data being binned
        zs (array): whatever value is being binned (i.e. elevation)

    Returns:
          dictionary with following keys for all gauges
            'xBinMedian' (array): xn by yn, median x-location value in each bin
            'yBinMedian' (array): xn by yn, median y-location value in each bin
            'zBinMedian' (array): xn by yn, median value in each bin of whatever variable is being "binned"
            'binCounts' (array): xn by yn, number of data points in each grid/bin
            'xBinVar' (array): xn by yn, variance of x-locations in each bin
            'yBinVar' (array): xn by yn, variance of y-locations in each bin
            'zBinVar' (array): xn by yn, variance of whatever variable is being binned
            'points' (array): for trouble shooting, the input x,y data being binned


    TODO:
    """
    import numpy as np
    points = np.vstack((xs, ys))
    points = points.T
    # bin example survey to the grid (use median to remove outliers)
    # grid bin size
    dx = np.median(np.diff(xn))
    dy = np.median(np.diff(yn))
    # grid edges
    xe = np.arange(xn[0] - dx / 2, xn[-1] + dx, dx)
    ye = np.arange(yn[0] - dy / 2, yn[-1] + dy, dy)
    # grid outline
    xg, yg = np.meshgrid(xe, ye)
    # yg, xg = np.meshgrid(ye, xe)
    # grid corners
    xgSW = xg[0:-1, 0:-1]
    ygSW = yg[0:-1, 0:-1]
    xgSE = xg[0:-1:, 1:]
    ygSE = yg[0:-1:, 1:]
    xgNW = xg[1:, 0:-1]
    ygNW = yg[1:, 0:-1]
    xgNE = xg[1:, 1:]
    ygNE = yg[1:, 1:]

    zBinMedian = np.empty((len(yn), len(xn)))
    zBinMedian[:] = np.nan
    xBinMedian = np.empty((len(yn), len(xn)))
    xBinMedian[:] = np.nan
    yBinMedian = np.empty((len(yn), len(xn)))
    yBinMedian[:] = np.nan

    zBinVar = np.empty((len(yn), len(xn)))
    zBinVar[:] = np.nan
    xBinVar = np.empty((len(yn), len(xn)))
    xBinVar[:] = np.nan
    yBinVar = np.empty((len(yn), len(xn)))
    yBinVar[:] = np.nan

    binCounts = np.zeros((len(yn), len(xn)))

    from collections import Counter
    for num, j in enumerate(yn):
        for num2, i in enumerate(xn):
            vertices = np.array([[xgSW[num, num2], ygSW[num, num2]], [xgSE[num, num2], ygSE[num, num2]],
                                 [xgNE[num, num2], ygNE[num, num2]], [xgNW[num, num2], ygNW[num, num2]]])
            p = path.Path(vertices)
            inside = p.contains_points(points)

            if any(inside):
                binValues = np.ma.MaskedArray.filled(zs[inside], np.nan)
                xSub = np.ma.MaskedArray.filled(xs[inside], np.nan)
                ySub = np.ma.MaskedArray.filled(ys[inside], np.nan)

                realValues = ~np.isnan(binValues)

                if any(realValues):
                    binValues[~realValues] = -999
                    goodValues = np.nonzero(binValues > -100)
                    obs = binValues[goodValues]
                    zBinMedian[num][num2] = np.nanmedian(binValues[goodValues])
                    xBinMedian[num][num2] = np.nanmedian(xSub[goodValues])
                    yBinMedian[num][num2] = np.nanmedian(ySub[goodValues])

                    zBinVar[num][num2] = np.nanvar(binValues[goodValues])
                    xBinVar[num][num2] = np.nanvar(xSub[goodValues])
                    yBinVar[num][num2] = np.nanvar(ySub[goodValues])
                    # print(len(obs))
                    binCounts[num][num2] = len(obs)#np.count_nonzero(~np.isnan(binValues))# Counter(goodValues) #np.count_nonzero(~np.isnan(goodValues))  # len(binValues)
                    del goodValues
                    del binValues

    output = dict()
    output['xBinMedian'] = xBinMedian
    output['yBinMedian'] = yBinMedian
    output['zBinMedian'] = zBinMedian
    output['binCounts'] = binCounts
    output['xBinVar'] = xBinVar
    output['yBinVar'] = yBinVar
    output['zBinVar'] = zBinVar
    output['points'] = points
    return output


def coarseBackground(x, y, z, cdx,cdy):
    """Need a coarse background DEM to start everything off

    Args:
        x (array): x-coordinates for spatial data being binned
        y (array): y-coordinates for spatial data being binned
        z (array): whatever value is being binned (i.e. elevation)

    Returns:
        xc (array): x-coordinates grid for coarser mesh
        yc (array): y-coordinates grid for coarser mesh
        zc (array): z-coordinates grid for coarser mesh
        xn (array): x-vector (input to meshgrid)
        yn (array): y-vector (input to meshgrid)

    TODO:
    """
    from scipy import interpolate

    yn = np.arange(y[0] - cdx, y[-1] + cdx, cdx)
    xn = np.arange(x[0] - cdy, x[-1] + cdy, cdy)
    xc, yc = np.meshgrid(xn, yn)
    f = interpolate.interp2d(x, y, z, kind='linear')
    zc = f(xn, yn)

    return xc, yc, zc, xn, yn


def scatterDEM(x, y, z, title, label='elevation (m)', cmap='YlGnBu'):
    import matplotlib.pyplot as plt
    fig3a, ax3a = plt.subplots(1, 1, figsize=(5, 5))
    sc3a = ax3a.scatter(x, y, c=z, cmap=cmap)
    cbar = plt.colorbar(sc3a, ax=ax3a)
    cbar.set_label(label)
    ax3a.set_title(title)

def pcolorDEM(x, y, z, title, label='elevation(m)'):
    import matplotlib.pyplot as plt
    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))
    pc2 = ax3.pcolor(x, y, z)
    cbar = plt.colorbar(pc2, ax=ax3)
    cbar.set_label(label)
    ax3.set_title(title)

class getGeoDatasets:
    """ A call that goes to each relevant morphological data set and outputs in a similar format.
    Ideally this will be used in a larger script identifying which files are recorded in some
    time/date window, and those files identified are passed to this.

    Returns:
        dictionary with the following:
            x: xFRF
            y: yFRF
            z: elevation

    TODO: getGeoDatasets has all been written with local files in the same folder as the scripts, needs to
        be expanded to include going to THREDDS (use getDataFRF)

    """
    def __init__(self, **kwargs):

        self.bathyFile = kwargs.get('bathyFile', None)
        self.clarisFile = kwargs.get('clarisFile', None)
        self.duneFile = kwargs.get('duneFile', None)
        self.pierFile = kwargs.get('pierFile', None)

    def getBathy(self):
        """Operates on a bathy file with transect data
        BUT: it throws out data way north of the FRF property (y=3000m)
        """
        bathy = Dataset(self.bathyFile)
        # get out the masked variables
        xs_bathy = bathy.variables['xFRF'][:]
        ys_bathy = bathy.variables['yFRF'][:]
        zs_bathy = bathy.variables['elevation'][:]
        ts_bathy = bathy.variables['time'][:]
        zs_bathy = np.ma.masked_where(ys_bathy > 3000, zs_bathy)
        ys_bathy = np.ma.masked_where(ys_bathy > 3000, ys_bathy)
        xs_bathy= np.ma.masked_where(ys_bathy > 3000, xs_bathy)
        output = dict()
        output['x'] = xs_bathy
        output['y'] = ys_bathy
        output['z'] = zs_bathy
        return output


    def getDune(self):
        """ Operates on Dune Lidar files
        """
        lidar = Dataset(self.duneFile)
        xs_lidar = lidar.variables['xFRF'][:]
        ys_lidar = lidar.variables['yFRF'][:]
        xs_lidar, ys_lidar = np.meshgrid(xs_lidar, ys_lidar)
        xs_lidar = xs_lidar.flatten()
        ys_lidar = ys_lidar.flatten()
        zs_lidar = lidar.variables['elevation'][:]
        zs_lidar = zs_lidar[0, :, :]
        zs_lidar = zs_lidar.flatten()
        ts_lidar = lidar.variables['time'][:]
        zs_lidar = np.ma.masked_where(zs_lidar < -900, zs_lidar)
        ys_lidar = np.ma.masked_where(zs_lidar < -900, ys_lidar)
        xs_lidar = np.ma.masked_where(zs_lidar < -900, xs_lidar)

        output = dict()
        output['x'] = xs_lidar
        output['y'] = ys_lidar
        output['z'] = zs_lidar

        return output


    def getPier(self):
        """Operates on pier lidar files
        TODO: Could fold this into a generic get lidar script, slowly collapsing to the same thing as the dune lidar
        """
        plidar = Dataset(self.pierFile)
        xs_plidar = plidar.variables['xFRF'][:]
        ys_plidar = plidar.variables['yFRF'][:]
        xs_plidar, ys_plidar = np.meshgrid(xs_plidar, ys_plidar)
        xs_plidar = xs_plidar.flatten()
        ys_plidar = ys_plidar.flatten()
        zs_plidar = plidar.variables['elevation'][:]
        zs_plidar = zs_plidar[0, :, :]
        zs_plidar = zs_plidar.flatten()
        ts_plidar = plidar.variables['time'][:]
        zs_plidar = np.ma.masked_where(zs_plidar < -900, zs_plidar)
        ys_plidar = np.ma.masked_where(zs_plidar < -900, ys_plidar)
        xs_plidar = np.ma.masked_where(zs_plidar < -900, xs_plidar)

        output = dict()
        output['x'] = xs_plidar
        output['y'] = ys_plidar
        output['z'] = zs_plidar

        return output


    def getClaris(self):
        """ Operates on Claris files, which are not yet on THREDDS, and may not be saved in the format/fashion this
        script is designed to handle. Currently operates on a .mat file using h5py. A transformation is applied to get
        everything to NC state plane and then to FRF coords with testbedutils.geoprocess
        BUT only keeps data between yFRF=-100 and 1400
        """

        with h5py.File(self.clarisFile, 'r') as f:
            for k in f.keys():
                print(k)
            x = f['grid/x'][:]
            y = f['grid/y'][:]
            z = f['grid/z'][:]

        rot = np.array([[0.933218541975915, -0.359309271954326], [0.359309271954326, 0.933218541975915]])

        points = np.vstack([x.flatten(), y.flatten()])  # , z.flatten()])

        rotated = np.matmul(rot, points)
        rotated = rotated.T

        NCx = rotated[:, 0] + 9.030235779999999e+05
        NCy = rotated[:, 1] + 2.710970920000000e+05
        FRF = geoprocess.FRFcoord(NCx, NCy)

        x_claris = FRF['xFRF']
        y_claris = FRF['yFRF']
        z_claris = z.flatten()

        del x, y, z
        indomain = np.where(np.logical_and(y_claris > -100, y_claris < 1400))

        y_claris = y_claris[indomain]
        x_claris = x_claris[indomain]
        z_claris = z_claris[indomain]

        output = dict()
        output['x'] = x_claris
        output['y'] = y_claris
        output['z'] = z_claris

        return output