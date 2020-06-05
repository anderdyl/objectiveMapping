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
    pc2 = ax3.pcolormesh(x, y, z)
    cbar = plt.colorbar(pc2, ax=ax3)
    cbar.set_label(label)
    ax3.set_title(title)