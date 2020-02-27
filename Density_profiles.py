#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from itertools import combinations
from seawater.eos80 import pres, dens, pden


class profile:
    """
    Class for a oceanic temperature and salinity profile.

    ...

    Attributes
    ----------
    H : float
        Horizontal scale. Default 4400.
    den0 : float
        Density reference value. Default 1029.
    U : float
        Horizontal speed scale. Default 0.1.
    g : float
        Gravity value. Default 9.8.
    times : ndarray (1D)
        Time values of the data.
    lat : ndarray 
        Latitude values of the data.
    lon : ndarray 
        Longitude values of the data.
    saline: ndarray
        Salinity values.
    temper: ndarray
        Temperature values.
    area : str
        Dataset area.
    date : str
        Dataset date.
    dataset : str
        Dataset name (without variable).
    path2data : str
        Path to the folder where the data is.
    """

    path2data = '/mnt/meom/workdir/alberta2/eNATL60/' +\
                'eNATL60-BLBT02-S/1h/profGULF-OSMO/'
    path2save = '/mnt/meom/workdir/martiene/params/'
    omega2 = 2*7.2921e-5
    a = 6.371e6
    L0 = 100
    N = 512
    Ekb = 5.e-4
    Re4 = 1000.
    tau0  = 1e-3
    DT = 5e-2
    tend  = 2000.
    dtout = 1.
    CFL   = 0.5
    

    def __init__(self, area, date, L=50000., H=4400., den0=1029., U=.1, g=9.8):
        self.H = H
        self.L = L
        self.den0 = den0
        self.U = U
        self.g = g
        self.area = area
        self.date = date
        self.dataset = 'eNATL60p' + area + '-BLBT02_' + date + '.1h'
        self.times, self.lat, self.lon, self.depths, self.saline, self.temper = self.get_values()


    def get_values(self):
        salname = self.path2data + self.dataset + '_vosaline.nc'
        temname = self.path2data + self.dataset + '_votemper.nc'
        with xr.open_dataset(salname) as data:
            sal = data['vosaline'].values
            tim = data.time_counter.values
            lat = data.nav_lat.values
            lon = data.nav_lon.values
            dep = data.deptht.values
        with xr.open_dataset(temname) as data:
            tem = data['votemper'].values
        ind = (dep < self.H)
        sal, tem, dep = sal[:,ind,:,:], tem[:,ind,:,:], dep[ind]
        return tim, lat, lon, dep, sal, tem


    def get_pressures(self):
        """
        Compute pressure values using seawater.eos80.pres

        New attributes
        --------------
        pressures : ndarray   
            Pressure values of the data.
        """
        self.pressures = pres(self.depths, np.mean(self.lat))


    def get_density(self):
        """
        Compute density values using seawater.eos80.dens

        New attributes
        --------------
        density : ndarray   
            Density values of the data.
        """
        if not hasattr(self, 'pressures'):
            self.get_pressures()
        sal = np.mean(self.saline, axis=(2, 3))
        tem = np.mean(self.temper, axis=(2, 3))
        self.density = dens(sal, tem, self.pressures)


    def get_pdensity(self):
        """
        Compute potential density values using seawater.eos80.pden

        New attributes
        --------------
        pdensity : ndarray   
            Potential density values of the data.
        """
        if not hasattr(self, 'pressures'):
            self.get_pressures()
        sal = np.mean(self.saline, axis=(2, 3))
        tem = np.mean(self.temper, axis=(2, 3))
        self.pdensity = pden(sal, tem, self.pressures)


    def plot_density(self, **kwards):
        """
        Plots mean density values.
        """
        if not hasattr(self, 'density'):
            self.get_density()
        mden = np.mean(self.density, axis=0)
        plt.plot(mden, self.depths, **kwards)
        plot_extras(r"Density ($kg\,m^{-3}$)", self.H)


    def plot_pdensity(self, **kwards):
        """
        Plots mean potential density values.
        """
        if not hasattr(self, 'pdensity'):
            self.get_pdensity()
        mden = np.mean(self.pdensity, axis=0)
        plt.plot(mden, self.depths, **kwards)
        plot_extras(r"Potential density ($kg\,m^{-3}$)", self.H)

        
    def make_partition(self, met, npar=1, show=True, save=False,
                            dep=(0, 5000), nlsep=1, p=2, **kwards):
        """
        Gives a discretization using the given method for the potential density.
        
        Arguments
        ---------
        met: ["gra", "max"]
            Name of the method to be used.
            max: Gives a discretization using the maximum mean potential density values difertence between layers.
                 The maximum is given computing a p-norm, i.e., sum((den_i+1-den_i)**p).
            gra: Gives a discretization using the maximum gradient in potential density.
        npar: int
            Number of partitions to be done. I.e., number of layers - 1. Default 1.
        show: bool
            If True shows the discretization.
        save: False pf str
            If not False saves the parameters for the discretization in the msqg model.
        dep: (float, float)
            The estimated maximum depth for which the result will be in the first layer.
            The estimated minimum depth for which the result will be in the last layer.
            Makes the algorithm faster. Otherwise use dep = (0, H) to compute all values.
        nlsep: int
            The minimum number of layers to compute the subdivision. Default 1.
            Bigger number will make the algorithm faster but may disturb the solution.
            Minimum value must be 1.
        p: int (Only max method)
            The value of the p-norm to maximize.
        
        Returns
        -------
        ind: ndarray (1D)
            Array of the limits of each layer.
        
        """
        if not hasattr(self, 'pdensity'):
            self.get_pdensity()
        mden = np.mean(self.pdensity, axis=0)
        # Defining minimum and maximum index to compute between
        if dep[0] <= self.depths[0]:
            imin = 1
        else:
            imin = np.min(np.where(self.depths >= dep[0]))
        if dep[1] >= self.H:
            imax = len(self.depths)
        else:
            imax = np.max(np.where(self.depths <= dep[1]))
        # Starts the method
        if met == "grad":
            ind = self._make_partition_grad(mden, npar, nlsep, (imin, imax))
        elif met=="max":
            ind = self._make_partition_max(mden, npar, nlsep,  (imin, imax), p)
        # Adding first and last index to ind
        n = len(self.depths)
        ind = np.insert(ind, (0, npar), [0, n]).astype(int)
        # Plotting
        if show: self._plot_dis(mden, ind, **kwards)
        # Saving parameters
        if save is not False: self._create_params_file(save, npar, mden, ind)


    def _make_partition_grad(self, mden, npar, nlsep, ilim):
        # Computes the density gradient
        dpden = np.diff(mden[ilim[0]:ilim[1]])/np.diff(self.depths[ilim[0]:ilim[1]])
        m = len(dpden)
        # initialize index and maximum number of loops
        ind = []
        while len(ind) < npar:
            # find the index of the maximum gradient value
            tind = np.argmax(dpden)
            ind.append(tind)
            # set used value and neightbours to 0
            dpden[max(tind-nlsep, 0):min(tind+nlsep, m)] = 0
            if np.all(dpden <= 0):
                raise ValueError('No convergence')
        # add minimum index and sort them
        ind = ilim[0] + np.sort(ind)
        return ind


    def _make_partition_max(self, mden, npar, nlsep, ilim, p):
        # Generate combinations of partitions
        comb = combinations(np.arange(ilim[0], ilim[1]), npar)
        maxval = 0
        for c in comb:
            if np.any(np.diff(c) <= nlsep):
                # if in a combination the layers are to close go to the next one
                continue
            # split and compute the mean values norm
            sp = np.split(mden, c)
            msp = np.array([np.mean(p) for p in sp])
            norm = np.sum(np.diff(msp)**p)
            if norm > maxval:
                # if the norm is bigger than the current value save the new combination
                maxval = norm
                ind = c
        return ind
    
    
    def _create_params_file(self, save, npar, mden, ind):
        params = self._get_params(mden, ind)
        outname = "params" + save + ".in"
        with open(self.path2save + outname, "w") as f:
            f.write("#!sh\n"+
                    "# " + outname + "\n"+
                    "# input parameter files\n"+
                    "# Generated with python\n\n"+
                    "# domain size\n"+
                    f"N  = {self.N}\n"+
                    f"nl = {npar+1}\n"+
                    f"L0 = {self.L0}\n\n"+
                    "# physical parameters\n"+
                    f"Rom   = -{params[2]}\n"+
                    f"Ekb   = {self.Ekb}\n"+
                    f"Re4   = {self.Re4}\n"+
                    f"beta  = {params[3]}\n"+
                    f"tau0  = {self.tau0}\n"+
                    "Fr = ["+
                    ",".join([f"{params[1][i]}" for i in range(npar)])+
                    "]\ndh = ["+
                    ",".join([f"{params[0][i]}" for i in range(npar+1)])+
                    "]\n\n"+
                    "# timestepping\n"+
                    f"DT = {self.DT}\n"+
                    f"tend  = {self.tend}\n"+
                    f"dtout = {self.dtout}\n"+
                    f"CFL   = {self.CFL}")

            
    def _plot_dis(self, den, ind, **kwards):
        for i in range(len(ind)-1):
            mdeni = np.mean(den[ind[i]:ind[i+1]])
            kwards = plot_extras(r"Potential density ($kg\,m^{-3}$)", self.H, **kwards)
            plt.vlines(mdeni, self.depths[ind[i]], self.depths[ind[i+1]-1], **kwards)

                
    def _get_params(self, mden, ind):
        ind[-1] = ind[-1] - 1
        dh = (self.depths[ind[1:]]- self.depths[ind[:-1]])
        h2 = .5 * (dh[1:]+dh[:-1])
        dh = dh/self.H
        sp = np.split(mden, ind[1:-1])
        msp = np.array([np.mean(p) for p in sp])
        dp = np.diff(msp)
        N = np.sqrt(self.g*dp/(h2*self.den0))
        Fr = self.U/(N*self.H)
        f = self.omega2*np.sin(np.mean(self.lat))
        Ro = self.U/(f*self.L)
        beta = self.omega2/self.a*np.cos(np.mean(self.lat))*self.L**2/self.U
        return (dh, Fr, Ro, beta)
    


def plot_extras(variable, H, **kwards):
    plt.plot([],[],**kwards)
    if 'label' in kwards:
        kwards.pop('label', None)
    plt.ylabel(r"Depth ($m$)")
    plt.xlabel(variable)
    plt.ylim(H, 0)
    return kwards

