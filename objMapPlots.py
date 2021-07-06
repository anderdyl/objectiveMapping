from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colors

class MidpointNormalize(colors.Normalize):
	"""	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	Examples:
        elev_min=-1000
        elev_max=3000
        mid_val=0
        plt.imshow(ras, cmap=cmap, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
        plt.colorbar()
        plt.show()
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def QAQCplot(ofname, xxFRFbackground, yyFRFbackground, fi, bathy, dune, pier, **kwargs):
    """
    
    Args:
        ofname:
        xxFRFbackground:
        yyFRFbackground:
        fi:
        bathy:
        dune:
        pier:
        **kwargs:
    Keyword Args:
         xbounds: cross-shore bounds for all plots (default=[0, 1000])
         ybounds: alonshore bounds for all plots (default=[-200,1500]
         figsize: figure size (default=[14, 8])
         diff: plot the differential
         
    Returns:

    """
    from scipy import spatial
    cmap = 'ocean_r'
    xbounds = kwargs.get('xbounds', [0, 1000])
    ybounds = kwargs.get('ybounds', [-200, 1500])
    figsize = kwargs.get('figSize', [14,8])
    diff = kwargs.get('diff', None)
    
    xFRFbackground, yFRFbackground = np.array(xxFRFbackground[0]), np.array(yyFRFbackground[:,1])
    # make kdtree for output
    gridOutTree = spatial.cKDTree(data=np.c_[np.array(xxFRFbackground.flatten()), np.array(yyFRFbackground.flatten())])
    ###############################
    # calculate bathy residuals
    duneData, pierData = False, False
    if bathy is not None:
        obsTree = spatial.cKDTree(data=np.c_[np.array(bathy['xFRF']), np.array(bathy['yFRF'])])
        bathyOut = obsTree.query_ball_tree(gridOutTree, 2)

        bathyGridMatched, bathyObsMatched, xyBathy = [], [], []
        for ii in range(bathy['elevation'].shape[0]):
            if len(bathyOut[ii]) > 0:
                xyBathy.append((bathy['xFRF'][ii], bathy['yFRF'][ii]))
                bathyGridMatched.append(fi.flatten()[bathyOut[ii]])
                bathyObsMatched.append(bathy['elevation'][ii])
        bathyGridMatched = np.squeeze(bathyGridMatched)
        bathyObsMatched = np.squeeze(bathyObsMatched)
        bathyResiduals = (bathyGridMatched[bathyGridMatched<0] - bathyObsMatched[bathyGridMatched<0])
    
    ## now look at pier
    if pier is not None:
        xx, yy = np.meshgrid(np.array(pier['xFRF']), np.array(pier['yFRF']))
        pierTree = spatial.cKDTree(data=np.c_[xx.flatten(), yy.flatten()])
        pierOut = pierTree.query_ball_tree(gridOutTree, 2)
        pierGrid_avg = np.nanmean(pier['elevation'], axis=0)
        pierGridMatched, pierObsMatched, xyPier = [], [], []
        for ii in range(pierGrid_avg.flatten().shape[0]):
            if len(pierOut[ii]) > 0 and not np.ma.is_masked(pierGrid_avg.flatten()[ii]):
                xyPier.append((xx.flatten()[ii], yy.flatten()[ii]))
                pierGridMatched.append(fi.flatten()[pierOut[ii]])
                pierObsMatched.append(pierGrid_avg.flatten()[ii])
        pierGridMatched = np.squeeze(pierGridMatched)
        pierObsMatched = np.squeeze(pierObsMatched)
        pierResiduals = (pierGridMatched - pierObsMatched)
        pierData = True
    # now look at Dune
    if dune is not None:
        xx, yy = np.meshgrid(np.array(dune['xFRF']), np.array(dune['yFRF']))
        duneTree = spatial.cKDTree(data=np.c_[xx.flatten(), yy.flatten()])
        duneOut = duneTree.query_ball_tree(gridOutTree, 2)
        duneGrid_avg = np.nanmean(dune['elevation'], axis=0)
        duneGridMatched, duneObsMatched, xyDune = [], [], []
        for ii in range(duneGrid_avg.flatten().shape[0]):
            if len(duneOut[ii]) > 0 and not np.ma.is_masked(duneGrid_avg.flatten()[ii]):
                xyDune.append((xx.flatten()[ii], yy.flatten()[ii]))
                duneGridMatched.append(fi.flatten()[duneOut[ii]])
                duneObsMatched.append(duneGrid_avg.flatten()[ii])
        duneGridMatched = np.squeeze(duneGridMatched)
        duneObsMatched = np.squeeze(duneObsMatched)
        duneResiduals = (duneGridMatched - duneObsMatched)
        duneData = True
    # get colorbar min/max for topo comparison
    try:
        topoCmin = min(np.nanmin(duneResiduals), np.nanmin(pierResiduals))
        topoCmax = max(np.nanmax(duneResiduals), np.nanmax(pierResiduals))
    except (NameError, UnboundLocalError):
        if duneData is True:
            topoCmin = np.nanmin(duneResiduals)
            topoCmax = np.nanmax(duneResiduals)
        if pierData is True:
            topoCmin = np.nanmin(pierResiduals)
            topoCmax = np.nanmax(pierResiduals)
    ############################################################################################
    ############################################################################################
    if diff is not None:
        plotSize= (2,7)
        figsize = [18,8]
    else:
        plotSize = (2,5)
    plt.figure(figsize=(figsize))
    ax1 = plt.subplot2grid(plotSize, (0,0), colspan=2)
    ax1.set_title('Residuals between survey and resultant grid')
    bb = ax1.pcolormesh(xFRFbackground, yFRFbackground, fi, cmap=cmap)
    ax1.contour(xFRFbackground, yFRFbackground, fi, '..',colors='k', levels=[-8, -6, -4.5, -3, -2, -1], linewidths=0.5)
    if bathy is not None:
        try:
            dd = ax1.scatter(np.array(xyBathy)[:, 0][bathyGridMatched<0], np.array(xyBathy)[:,1][bathyGridMatched<0],
                    c=bathyResiduals, cmap='RdBu',
                    norm=MidpointNormalize(midpoint=0, vmin=bathyResiduals.min(),vmax=bathyResiduals.max()), s=3)
        except:
            pass
    
    ax1.plot([0,515], [500,500], 'k-', lw=5)
    try:
        cbar = plt.colorbar(dd, ax=ax1)
        cbar.set_label('grid residual from measurement')
    except:
        cbar = plt.colorbar(bb, ax=ax1)
        cbar.set_label('depth [m]')
    ax1.set_ylim(ybounds)
    ax1.set_xlim(xbounds)
    ###############
    ax2 = plt.subplot2grid(plotSize,(0,2))
    ax2.plot([-10, 5], [-10,5], 'k--')
    try:
        ax2.plot(bathyGridMatched, bathyObsMatched, 'r.')
        rmse = np.sqrt((bathyResiduals**2).sum()/len(bathyResiduals))
        ax2.text(-8, -11, s='RMSE {:.2f}'.format(rmse))
    except NameError:
        ax2.text(-5, -2.5, 'No Bathy')
        rmse = -999
    ax2.set_xlabel('grid depth [m]')
    ax2.set_ylabel('survey depth [m]')
    ###############
    ax3 = plt.subplot2grid(plotSize, (0,3), colspan=2)
    for idx in np.argwhere(np.in1d(yFRFbackground, [0, 200, 400, 600, 800, 1000])).squeeze():
       ax3.plot(xFRFbackground, fi[idx], label="y={}".format(yFRFbackground[idx]))
    ax3.legend(fancybox=True, framealpha=0.25, fontsize=9, ncol=2, loc='upper right')
    ax3.set_xlabel('xFRF')
    ax3.set_ylabel('elevation [m]')
    ###############
    ax4 = plt.subplot2grid(plotSize, (1,0), colspan=2)
    ax4.set_title('Residuals between lidar and resultant grid')
    b = ax4.pcolormesh(xFRFbackground, yFRFbackground, fi, cmap=cmap)
    CS = ax4.contour(xFRFbackground, yFRFbackground, fi, '..',colors='k', levels=[-2, -1, 0, 1, 2],
                linewidths=0.5)
    plt.clabel(CS, fontsize=10, fmt='%d') # ,fmt = '%1.0f'

    if pier is not None:
        d = ax4.scatter(np.array(xyPier)[:, 0],
                    np.array(xyPier)[:,1],
                    c=pierResiduals, cmap='RdBu',
                    norm=MidpointNormalize(midpoint=0, vmin=topoCmin,vmax=topoCmax), s=1)
    if dune is not None:
        d = ax4.scatter(np.array(xyDune)[:, 0],
                    np.array(xyDune)[:,1],
                    c=duneResiduals, cmap='RdBu',
                    norm=MidpointNormalize(midpoint=0, vmin=topoCmin,vmax=topoCmax), s=1)

    ax4.plot([0,515], [500,500], 'k-', lw=5)
    try:
        cbar = plt.colorbar(d, ax=ax4)
        cbar.set_label('grid residual from measurement')
    except NameError:
        cbar = plt.colorbar(b, ax=ax4)
        cbar.set_label('depth [m]')
    ax4.set_ylim(ybounds)
    ax4.set_xlim([0,200])
    #############
    ax5 = plt.subplot2grid(plotSize, (1,2))
    if pier is not None:
        ax5.plot(pierGridMatched, pierObsMatched, 'b.', ms=2, label='pier')
        pierRmse = np.sqrt((pierResiduals**2).sum()/len(pierResiduals))
        ax5.text(-1, 9, s='RMSE {:.2f}'.format(pierRmse), c='b')
    else:
        pierRmse = -999
    if dune is not None:
        ax5.plot(duneGridMatched, duneObsMatched, 'g.', ms=2, label='dune')
        duneRmse = np.sqrt((duneResiduals**2).sum()/len(duneResiduals))
        ax5.text(-1, 7, s='RMSE {:.2f}'.format(duneRmse), c='g')
    else:
        duneRmse = -999
    
    ax5.plot([-1, 10], [-1, 10], 'k--')
    ax5.legend(fontsize=9, loc='lower right')
    ax5.set_xlabel('grid elev. [m]')
    ax5.set_ylabel('topo elev. [m]')
    #############
    ax6 = plt.subplot2grid(plotSize, (1,3), colspan=2)
    for idx in np.argwhere(np.in1d(xFRFbackground, [50, 100, 200, 300, 450, 600])).squeeze():
        ax6.plot(yFRFbackground, fi[:, idx], label="x={}".format(xFRFbackground[idx]))
    ax6.legend(fancybox=True, framealpha=0.25, fontsize=9, ncol=2, loc='upper right')
    ax6.set_xlabel('yFRF')
    ax6.set_xlim(ybounds)
    ax6.set_ylabel('elevation [m]')
    if diff is not None:
        ax8 = plt.subplot2grid(plotSize, (0, 5), colspan=2, rowspan=2)
        surf = ax8.pcolor(xFRFbackground, yFRFbackground, diff, cmap='RdBu', norm=MidpointNormalize(midpoint=0))
        ax8.contour(xFRFbackground, yFRFbackground, fi,'..',colors='k', levels=[-8, -6, -4.5,  -2, 0],
                    linewidths=0.5)
        cbar = plt.colorbar(surf, ax=ax8)
        ax8.set_xlim(xbounds)
        ax8.set_xlabel('xFRF')
        ax8.set_ylabel('yFRF')
        ax8.set_ylim(ybounds)
        ax8.set_title('Difference map\n[reds/blues accretion/erosions]')
        cbar.set_label('change [m]')
        plt.text(200, 200, "changed {} of {}\ncells for {:.1f}% change".format((diff!=0).sum(), np.size(diff),
                                                                               ((diff!=0).sum())/np.size(diff)*100))
    plt.tight_layout(w_pad=0.1)
    plt.savefig(ofname)
    plt.close()
    return pierRmse, duneRmse, rmse
    

def binnedDataPlot(ofname, binned, xc, yc, id, stdErr, zFluc, **kwargs):
    """Creates a 4 panel preprocessing step plot.
    
    Args:
        ofname: output file name
        binned: spatially binned output 'xBinMedian' 'yBinMedian' 'zBinMedian' 'binCounts'
        xc: whole domain of x values
        yc: whole domain of y values
        id: domain subset indicies
        stdErr: standardized error
        zFluc: fluctuation of elevation data in the bin
        **kwargs:
            'fontSize': fontsize (default=12)
            'cmap': colormap (default='YlGnBu')
            'leaveOpen': if true, function will not close plot (default=False)
    Returns:
        None
    """
    fs = kwargs.get('fontSize', 12)
    cmap = kwargs.get('cmap', 'YlGnBu')
    leaveOpen = kwargs.get('leaveOpen', False)
    figSize = kwargs.get('figSize', (12,12))
    #######################################################################################
    fig = plt.figure(figsize=figSize)
    ax1 = plt.subplot2grid((2,2), (0,0))
    pc1 = ax1.scatter(binned['xBinMedian'], binned['yBinMedian'], c=binned['zBinMedian'], cmap=cmap)
    cbar = plt.colorbar(pc1, ax=ax1)
    ax1.set_title("Binned Survey", fontsize=fs)
    cbar.set_label('elevation (m)')
    
    # pcolorDEM(x=xc, y=yc, z=binned['binCounts'], title="Binned observations", label='data points per bin')
    ax2 = plt.subplot2grid((2,2), (0,1), sharex=ax1, sharey=ax1)
    if (binned['binCounts'] > 1000).any():
        pc2 = ax2.pcolormesh(xc, yc, binned['binCounts'], norm=colors.LogNorm())
    else:
        pc2 = ax2.pcolormesh(xc, yc, binned['binCounts'])
    cbar = plt.colorbar(pc2, ax=ax2)
    cbar.set_label('data points per bin', fontsize=fs)
    ax2.set_title("Binned observations", fontsize=fs)

    
    # x=xc[id], y=yc[id], z=stdErr, title='standard error [m]', label='standard error [m]'
    ax3 = plt.subplot2grid((2,2), (1,0), sharex=ax1, sharey=ax1)
    pc3 = ax3.scatter(xc[id], yc[id], c=stdErr, cmap=cmap)
    cbar = plt.colorbar(pc3, ax=ax3)
    ax3.set_title("standard error [m]", fontsize=fs)
    cbar.set_label('standard error [m]', fontsize=fs)
    
    ax4 = plt.subplot2grid((2,2), (1,1), sharex=ax1, sharey=ax1)
    pc4 = ax4.scatter(xc[id], yc[id], c=zFluc, cmap='RdBu', norm=MidpointNormalize(midpoint=0))
    cbar = plt.colorbar(pc4, ax=ax4)
    ax4.set_title("Difference btw Median\n binned obs and previous map", fontsize=fs)
    cbar.set_label('elevation difference [m]', fontsize=fs)
    # ax4.set_xlim([xc[id].min(), xc[id].max()])
    span = yc[id].max() - yc[id].min()
    ax4.set_ylim([yc[id].min()-0.1*span, yc[id].max()+0.1*span])
    plt.tight_layout()
    plt.savefig(ofname)
    if leaveOpen is False:
        plt.close()
        
def postProcessedGridPlot(ofname, xCoarse, yCoarse, nmseest, mapfluc, mapz, goodi,  **kwargs):
    """Creates a 4 panel post-processing step plot.
    
    Args:

        ofname: output file name
        xCoarse:
        yCoarse:
        nmseest:
        mapfluc:
        mapz:
        goodi: indicies of good values a mapfluc
        **kwargs:
            'fontSize': fontsize (default=12)
            'cmap': colormap (default='YlGnBu')
            'leaveOpen': if true, function will not close plot (default=False)
    Returns:
        None
    """
    fs = kwargs.get('fontSize', 12)
    cmap = kwargs.get('cmap', 'YlGnBu')
    leaveOpen = kwargs.get('leaveOpen', False)
    figSize = kwargs.get('figSize', (12,12))
    #######################################################################################
    fig = plt.figure(figsize=figSize)
    ax1 = plt.subplot2grid((2,2), (1,0))
    sc1 = ax1.pcolormesh(xCoarse, yCoarse, nmseest, cmap=cmap)
    cbar = plt.colorbar(sc1, ax=ax1)
    cbar.set_label('elevation [m]')
    ax1.set_title('NMSE estimate')
    
    ax2 = plt.subplot2grid((2,2), (0,1), sharex=ax1, sharey=ax1)
    sc2 = ax2.pcolormesh(xCoarse, yCoarse, mapfluc, cmap='bwr', norm=MidpointNormalize(midpoint=0))
    cbar = plt.colorbar(sc2, ax=ax2)
    cbar.set_label('elevation [m]')
    ax2.set_title('mapped perturbations')
    
    ax3 = plt.subplot2grid((2,2), (0,0), sharex=ax1, sharey=ax1)
    sc3 = ax3.scatter(xCoarse[goodi], yCoarse[goodi], c=mapfluc[goodi], s=100, norm=MidpointNormalize(midpoint=0),
                                                                               cmap='bwr')
    cbar = plt.colorbar(sc3, ax=ax3)
    cbar.set_label('elevation [m]')
    ax3.set_title('mapped perturbations (only good values))')
    
    ax4 = plt.subplot2grid((2,2), (1,1), sharex=ax1, sharey=ax1)
    sc4 = ax4.pcolormesh(xCoarse, yCoarse, mapz, cmap='ocean_r')
    cbar = plt.colorbar(sc4, ax=ax4)
    cbar.set_label('elevation [m]')
    ax4.set_title('final mapped elevation survey [m]')
    span = yCoarse[goodi].max()-yCoarse[goodi].min()
    ax4.set_ylim([yCoarse[goodi].min()-0.1*span, yCoarse[goodi].max()+0.1*span])
    
    plt.tight_layout()
    plt.savefig(ofname)
    if leaveOpen is False:
        plt.close()
    
def scatterDEM(x, y, z, title, label='elevation (m)', cmap='YlGnBu', **kwargs):
    """ make a scatter of values.

    Args:
        x:
        y:
        z:
        title:
        label:
        cmap: colormap (default='YlGnBu')

        **kwargs:
            ofname: outputfile name to save
            'fontSize': fontsize (default=12)
            'leaveOpen': if true, function will not close plot (default=False)
            'xBounds': limit x axes in FRF coordinates
            'yBounds': limit y axes in FRF coordinates
            
    Returns:

    """
    fs = kwargs.get('fontSize', 12)
    leaveOpen = kwargs.get('leaveOpen', False)
    ofname = kwargs.get('ofname', False)
    ########################################
    fig3a, ax3a = plt.subplots(1, 1, figsize=(5, 5))
    sc3a = ax3a.scatter(x, y, c=z, cmap=cmap)
    cbar = plt.colorbar(sc3a, ax=ax3a)
    cbar.set_label(label, fontsize=fs)
    ax3a.set_title(title, fontsize=fs)
    
    ####################################
    if ofname is not False:
        plt.savefig(ofname)
    if leaveOpen is False:
        plt.close()


def pcolorDEM(x, y, z, title, label='elevation(m)'):
    import matplotlib.pyplot as plt
    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))
    pc2 = ax3.pcolormesh(x, y, z)
    cbar = plt.colorbar(pc2, ax=ax3)
    cbar.set_label(label)
    ax3.set_title(title)