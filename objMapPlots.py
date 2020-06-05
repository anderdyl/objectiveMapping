from matplotlib import pyplot as plt

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
    ax2 = plt.subplot2grid((2,2), (0,1))
    pc2 = ax2.pcolormesh(xc, yc, binned['binCounts'])
    cbar = plt.colorbar(pc2, ax=ax2)
    cbar.set_label('data points per bin', fontsize=fs)
    ax2.set_title("Binned observations", fontsize=fs)
    
    # x=xc[id], y=yc[id], z=stdErr, title='standard error [m]', label='standard error [m]'
    ax3 = plt.subplot2grid((2,2), (1,0))#, sharex=ax1, sharey=ax1)
    pc3 = ax3.scatter(xc[id], yc[id], c=stdErr, cmap=cmap)
    cbar = plt.colorbar(pc3, ax=ax3)
    ax3.set_title("standard error [m]", fontsize=fs)
    cbar.set_label('standard error [m]', fontsize=fs)
    
    ax4 = plt.subplot2grid((2,2), (1,1), sharex=ax1, sharey=ax1)
    pc4 = ax4.scatter(xc[id], yc[id], c=zFluc, cmap='RdBu')
    cbar = plt.colorbar(pc4, ax=ax4)
    ax1.set_title("Difference btw Median\nObs and previous map", fontsize=fs)
    cbar.set_label('elevation fluctuation [m] of binned data', fontsize=fs)

    plt.tight_layout()
    plt.savefig(ofname)
    if leaveOpen is False:
        plt.close()

def scatterDEM(x, y, z, title, label='elevation (m)', cmap='YlGnBu', **kwargs):
    """ make a scatter of binned values

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
    Returns:

    """
    fs = kwargs.get('fontSize', 12)
    leaveOpen = kwargs.get('leaveOpen', False)
    ofname = kwargs.get('ofname', False)
    fig3a, ax3a = plt.subplots(1, 1, figsize=(5, 5))
    sc3a = ax3a.scatter(x, y, c=z, cmap=cmap)
    cbar = plt.colorbar(sc3a, ax=ax3a)
    cbar.set_label(label, fontsize=fs)
    ax3a.set_title(title, fontsize=fs)
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