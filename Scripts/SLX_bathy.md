---
layout: page
title: Script Yeray
---

## West mediteranean bathymetry map

From Stephanie (nov 2019)

### My code:

- Imports 

  (Note that that i import my own customed library where the actual plot code is ```import lib_medwest60 as slx```. See this library below.)

```python
## standart libraries
import os,sys
import numpy as np

from scipy.signal import argrelmax
from scipy.stats import linregress

# xarray
import xarray as xr

# plot
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean

# my custom plotting tools are loaded from this lib file
import lib_medwest60 as slx

%matplotlib inline 
```



* Read data: nav_lon, nav_lat and bathy from a netcdf file (standard bathy file from nemo):

* ```python
  nav_lat,nav_lon,bathy = slx.readonlybathy()
  ```

  

* Plot and save in png:

* ```python
  gstyle='darkstyle'
  plto = "bathy_pix_grid"    
  diro='/Users/leroux/DATA/RES/MEDWEST60_RES/'
  tlabel='bathymetry (m)'
  
  #========== regional limits of MED map =========
  xlim=(-6,11)
  ylim=(34.8,45.2)
  
  
  #====================================
  # data to plot 
  data2plot = bathy
  data2plot = data2plot.where(bathy!=0.,-999.)
  data2plot = data2plot.fillna(-999.)
  data2plot = data2plot.where(data2plot!=-999.)
  
  namo = plto+'.png'
  
  
  #========= Plot settings ===============
  levbounds=[0.,3200.,20]
  
  # customed colormap
  cmap,norm = slx.mycolormap(levbounds,cm_base=cmocean.cm.ice_r,cu='w',co='k')
  
  #========= PLOT ===============
  fig3,(ax) = plt.subplots(1, 1, figsize=[16, 12],facecolor='w')
  
  # main plot
  cs,ax = slx.plotmapMEDWEST_gp(fig3,ax,data2plot,cmap,norm,plto=plto,gridpts=True,gridptsgrid=True,fakegrid=False,gstyle=gstyle)
  
  # add date or anytitle
  tcolordate="#848484"
  tsizedate=14
  tdate=""
  #plt.annotate(tdate,xy=(15,775),xycoords='data', color=tcolordate,size=tsizedate)
  
  # add colorbar
  cb = slx.addcolorbar(fig3,cs,ax,levbounds,levincr=10,tformat="%.0f",tlabel=tlabel,facmul=1,orientation='horizontal',tc='w')
  
  
  # display 
  plt.show()
  
  # Save fig in png, resolution dpi    
  slx.saveplt(fig3,diro,namo,dpifig=400)
  ```



#### My plot library:

(Import with: ``` import lib_medwest60 as slx```)

```python
#!/usr/bin/env  python
#=======================================================================
"""
Stephanie Leroux
Collection of sy "customed" tools related to  MEDWEST60 analysis...
"""
#=======================================================================


## standart libraries
import os,sys
import numpy as np

from scipy.signal import argrelmax
from scipy.stats import linregress

# xarray
import xarray as xr

# plot
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean


# custom tools
import lib_medwest60


def readonlybathy(bathyfilepath='/Users/leroux/DATA/MEDWEST60_DATA/MEDWEST60-I/MEDWEST60_Bathymetry_v3.3.nc4'):
		'''def readonlybathy(bathyfilepath='/Users/leroux/DATA/MEDWEST60_DATA/MEDWEST60-I/MEDWEST60_Bathymetry_v3.3.nc4')'''  

    bathy =  xr.open_dataset(bathyfilepath)["Bathymetry"]
    # longitude
    nav_lon = xr.open_dataset(bathyfilepath)['nav_lon']
    # latitude
    nav_lat = xr.open_dataset(bathyfilepath)['nav_lat']
    
    return nav_lat,nav_lon,bathy
    
               
def plotmapMEDWEST_gp(fig3,ax,data2plot,cmap,norm,plto='tmp_plot',gridpts=True,gridptsgrid=False,gridinc=200,gstyle='lightstyle'): 
    cs  = ax.pcolormesh(data2plot,cmap=cmap,norm=norm)

    #ax = plt.gca()
    # Remove the plot frame lines. 
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False)  

    ax.tick_params(axis="both", which="both", bottom="False", top="False",  
                labelbottom="False", labeltop='False',left="False", right="False", labelright="False",labelleft="False")  

    
    if gridpts:
    # show gridpoint on axes
        ax.tick_params(axis="both", which="both", bottom="False", top="False",  
                labelbottom="True", labeltop='False',left="False", right="False", labelright="False",labelleft="True")  
        plto = plto+"_wthgdpts"

    if gridptsgrid:
        lstylegrid=(0, (5, 5)) 
        if (gstyle=='darkstyle'):
            cmap.set_bad('#424242')
            lcolorgrid='w'#"#585858" # "#D8D8D8"
            tcolorgrid='#848484'#"#848484"
            
        if (gstyle=='ddarkstyle'):
            cmap.set_bad('#424242')
            lcolorgrid='w'#"#585858" # "#D8D8D8"
            tcolorgrid='w'#'#848484'#"#848484"
        if (gstyle=='lightstyle'):
            cmap.set_bad('w')
            lcolorgrid="#585858" # "#D8D8D8"
            tcolorgrid='#848484'#"#848484"            

        lalpha=0.2
        lwidthgrid=1.
        #ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(gridinc))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(gridinc))   
        ax.tick_params(axis='x', colors=tcolorgrid)
        ax.tick_params(axis='y', colors=tcolorgrid)
        ax.grid(which='major',linestyle=lstylegrid,color=lcolorgrid,alpha=lalpha,linewidth=lwidthgrid)
        ax.axhline(y=1.,xmin=0, xmax=883,zorder=10,color=lcolorgrid,linewidth=lwidthgrid,linestyle=lstylegrid,alpha=lalpha )
    
    return cs,ax

def saveplt(fig,diro,namo,dpifig=300):
    fig.savefig(diro+namo, facecolor=fig.get_facecolor(), edgecolor='none',dpi=dpifig,bbox_inches='tight', pad_inches=0)
    plt.close(fig) 

def mycolormap(levbounds,cm_base='Spectral_r',cu='w',co='k'):
    lmin = levbounds[0]
    lmax = levbounds[1]
    incr = levbounds[2]
    levels = np.arange(lmin,lmax,incr)
    nice_cmap = plt.get_cmap(cm_base)
    colors = nice_cmap(np.linspace(0,1,len(levels)))[:]
    cmap, norm = from_levels_and_colors(levels, colors, extend='max')
    cmap.set_under(cu)
    cmap.set_over(co)
    return cmap,norm


def addcolorbar(fig,cs,ax,levbounds,levincr=1,tformat="%.2f",tlabel='',shrink=0.45,facmul=1.,orientation='vertical',pad=0.03,tc='k'):
    lmin = levbounds[0]
    lmax = levbounds[1]
    incr = levbounds[2]
    levels = np.arange(lmin,lmax,incr)
    cblev = levels[::levincr]
    
    if orientation =='horizontal':
        axins1 = inset_axes(ax,
                        height="15%",  # height : 5%
                            width="50%",
                        loc='lower right',
                        bbox_to_anchor=(0.08, 0.1,0.9,0.2),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
        
    if orientation =='vertical':
        axins1 = inset_axes(ax,
                        height="50%",  # height : 5%
                            width="2%",
                        loc='center left',
                       borderpad=2)

    cb = fig.colorbar(cs,cax=axins1,pad=pad,
                                    extend='both',                   
                                    ticks=cblev,
                                    spacing='uniform',
                                    orientation=orientation,
                                    )
    
    new_tickslabels = [tformat % i for i in cblev*facmul]
    cb.set_ticklabels(new_tickslabels)
    cb.ax.set_xticklabels(new_tickslabels, rotation=70,size=10,color=tc)
    cb.ax.tick_params(labelsize=10,color=tc) 
    cb.set_label(tlabel,size=14,color=tc)
    
    
    return cb,axins1

def textunit(varname):
    if varname=='SST':
        suffix=" (ºC)"
    if varname=='SSH':
        suffix=" (m)"
    if varname=='curloverf':
        suffix=""
    return suffix

```

