import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt

drifters = pd.read_csv(
    "/Users/hart-davis/Desktop/interpolated_gld.20180719_014717_time.txt",
    sep='\t')

vel = np.array(drifters.speed / 100)
vel[(vel >= 8)] = np.nan
ymd = drifters.date  # ymd stands for year month day
t = drifters.time
lonn = drifters.lon
latt = drifters.lat
ve = drifters.ve.values / 100
vn = drifters.vn.values / 100


def drifter_plotting3d(
        x,
        y,
        c,
        ctype=None,
        style=None,
        cmap=plt.cm.Spectral_r,
        level=None,
        vmin=None,
        vmax=None,
        bins=None,
        res='110m',
        proj=ccrs.Mercator(),
        cbar=False):
    """    x : longitudinal coordinate of  point/s
    y : latitudinal  coordinate of point/s
    c : property of coordinate. e.g. velocity at x,y.

    ctype: is the structure of x,y and c. If x,y and c are one-d arrays then ctype = None. [default]
              If x,y,c are in 2-d arrays, then ctype = two_dim

    style: default = None, which is a scatter plot. Options are scatter, pcolormesh, contourf. This can be
           developed and will be upgraded.

    cmap: color map for c. Default is plt.cm.Spectral_r

    level: For contour plots for the number of contours to be used.

    vmin/vmax: Minimum and maximum values to illustrate in plots.

    bins: The bin value for pcolormesh.

    res: Default is 110m. Can be set as 10m ,50n and 110m.

    proj: Projection to be used in cartopy. Options available in cartopy.
          Land and States are plotted by default."""

    print "Plot Set as : "
    land = cfeature.NaturalEarthFeature('physical', 'land', res,
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
    states = cfeature.NaturalEarthFeature(
        'cultural',
        name='admin_1_states_provinces_lines',
        scale=res,
        facecolor='none')
    proj = proj
    res = res

    fig = plt.figure(figsize=(20, 7.7))
    ax = plt.subplot(111, projection=proj)
    pc = ccrs.PlateCarree()

    extent = [x.min(), x.max(), y.min(), y.max()]
    ax.set_extent(extent, pc)

    ax.add_feature(land, facecolor='slategray', zorder=4)
    ax.coastlines(resolution=res)
    c = c[:len(x)]
    if style is None:
        print "None"
        ax.plot(x, y, marker='.', ms=0.2, c='k', lw=0, transform=pc)
        ax.plot(x, y, marker='.', ms=0.1, c='blue', lw=0, transform=pc)

    elif style == 'scatter':
        print "Scatter"
        ax.scatter(
            x,
            y,
            c=c,
            s=1,
            transform=pc,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax)
        gridxt, gridyt, velo = x, y, c
    elif style == 'pcolormesh':
        print "Pcolormesh"
        if ctype is None:
            if bins is None:
                gridxt = np.linspace(x.min(), x.max(), (x.max() - x.min()) / 2)
                gridyt = np.linspace(y.min(), y.max(), (y.max() - y.min()) / 2)
            elif bins == bins:
                gridxt = np.linspace(x.min(), x.max(), bins[0])
                gridyt = np.linspace(y.min(), y.max(), bins[1])

            longitudes, latitudes = np.meshgrid(gridxt, gridyt, sparse=True)
            #longitudes = np.transpose(longitudes)
            longitudes = np.squeeze(longitudes)
            latitudes = np.squeeze(latitudes)
            indxcheck = np.zeros([len(longitudes), len(latitudes)])
            vel = np.zeros([len(longitudes), len(latitudes)])
            # loop through the grid, find lon/lat values for each 0.25 grid
            # cell and average corresponding velocities
            for i in range(0, len(longitudes) - 1, 1):
                for j in range(0, len(latitudes) - 1, 1):
                    indx = np.where((x[:len(c)] >= longitudes[i])
                                    & (x[:len(c)] < longitudes[i + 1])
                                    & (y[:len(c)] >= latitudes[j])
                                    & (y[:len(c)] < latitudes[j + 1]))[0]

                    indxcheck[i, j] = len(indx)
                    # if no drifters found put a nan in the grid cell
                    if indx.size == 0:
                        vel[i, j] = 0
                    else:
                        vel[i, j] = np.mean(c[indx])
            velo = np.transpose(vel)
            velo = np.ma.masked_where(np.isnan(velo), velo)
            ss = ax.pcolormesh(
                gridxt,
                gridyt,
                velo,
                transform=pc,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax)
            plt.colorbar(ss)
        elif ctype == two_dim:
            ss = ax.pcolormesh(
                x,
                y,
                c,
                transform=pc,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax)
    elif style == 'contour':
        print "Contour"
        if ctype is None:

            gridxt = np.linspace(x.min(), x.max(), (x.max() - x.min()) / 2)
            gridyt = np.linspace(y.min(), y.max(), (y.max() - y.min()) / 2)
            longitudes, latitudes = np.meshgrid(gridxt, gridyt, sparse=True)
            #longitudes = np.transpose(longitudes)
            longitudes = np.squeeze(longitudes)
            latitudes = np.squeeze(latitudes)
            indxcheck = np.zeros([len(longitudes), len(latitudes)])
            vel = np.zeros([len(longitudes), len(latitudes)])
            # loop through the grid, find lon/lat values for each 0.25 grid
            # cell and average corresponding velocities
            for i in range(0, len(longitudes) - 1, 1):
                for j in range(0, len(latitudes) - 1, 1):
                    indx = np.where((x[:len(c)] >= longitudes[i])
                                    & (x[:len(c)] < longitudes[i + 1])
                                    & (y[:len(c)] >= latitudes[j])
                                    & (y[:len(c)] < latitudes[j + 1]))[0]

                    indxcheck[i, j] = len(indx)
                    # if no drifters found put a nan in the grid cell
                    if indx.size == 0:
                        vel[i, j] = 0
                    else:
                        vel[i, j] = np.mean(c[indx])
            velo = np.sqrt(vel**2)
            velo = np.transpose(velo)
            velo = np.ma.masked_where(np.isnan(velo), velo)
            ss = ax.contourf(
                gridxt,
                gridyt,
                velo,
                transform=pc,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                levels=np.linspace(
                    vmin,
                    vmax,
                    level))
            ass = ax.contour(
                gridxt,
                gridyt,
                velo,
                transform=pc,
                colors='black',
                vmin=vmin,
                vmax=vmax,
                levels=np.linspace(
                    vmin,
                    vmax,
                    level))
        elif ctype == two_dim:
            ss = ax.contourf(x, y, c, transform=pc, cmap=cmap)
            ass = ax.contour(x, y, c, transform=pc, colors='black')

        if cbar:
            cbar = plt.colorbar(ss)
            cbar.vmax = vmax
    return gridxt, gridyt, velo


def k_jk(x, y, ve, vn):

    lonn = x
    latt = y
    ve = ve
    vn = vn
    k_ = [0]
    la = []
    lo = []
    kx = [0]
    ky = [0]
    for l in range(0, len(np.unique(drifters.id)) - 1):
        indxx = np.where(drifters.id == np.unique(drifters.id)[l])[0]

        vel_e = np.array(ve[indxx])
        vel_n = np.array(vn[indxx])
        lon = np.array(lonn[indxx])
        lat = np.array(latt[indxx])
        vj = vel_e - np.nanmean(vel_e)
        vk = vel_n - np.nanmean(vel_n)

        dxx = []
        dyy = []
        for x in range(1, len(lat) - 1):
            dlat = (lat[x + 1] - lat[x - 1]) * 1.11e2 * 1000
            dlon = (lon[x + 1] - lon[x - 1]) * 1.11e2 * \
                np.cos(lat[x] * np.pi / 180.) * 1000
            dxx.append(dlon)
            dyy.append(dlat)
            lo.append(lon[x])
            la.append(lat[x])
        dx = dxx - np.nanmean(dxx)
        dy = dyy - np.nanmean(dyy)

        kjk = -(vj[1:-1] * dy)
        kkj = -(vk[1:-1] * dx)

        k = (kjk + kkj) / 2
        k_.extend(k)
        kx.extend(kjk)
        ky.extend(kkj)
    kjk = np.array(k_)
    loo = np.array(lo)
    laa = np.array(la)
    kx = np.array(kx)
    ky = np.array(ky)
    kjk = np.sqrt(kjk**2)
    return loo, laa, kx, ky, kjk


loo, laa, kx, ky, kjk = k_jk(lonn, latt, ve, vn)

xlonn, xlatt, eddyy = drifter_plotting3d(loo, laa, kjk, style='pcolormesh',
                                         bins=(100, 100), vmin=0,
                                         vmax=23000, level=5)
