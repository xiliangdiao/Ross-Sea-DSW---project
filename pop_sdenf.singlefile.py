# -*- coding: utf-8 -*
########################################################################################################################
### This code is originally from S. Yeager et al.,"An outsized role for the Labrador Sea in the multidecadal variability 
### of the Atlantic overturning circulation".Sci. Adv. 7, eabh3592 (2021).
### However, in our iHESP dataset, some variables are output only in the monthly dataset, while some other important
### variables are output only in the daily dataset. Xiliang Diao modified this original code to deal with this problem.
### (December21, 2023)  
########################################################################################################################

import xarray as xr               #netcdf multidim reading/writing/manipulation
import glob                       #globbing
import numpy as np                #numerics
import os                         #operating system commands
import subprocess
import time as timer
import pop_tools
import sys


time1=timer.time()

fin=sys.argv[-1]

#fin222 = fin[:-10] + 'nday1.' + fin[-10:-3] +'-01.mon.nc'
fin222 = fin[:-10] + 'nday1.' + fin[-10:-3] +'-01.nc'
print(fin222) 
#fin='/data3/model/cesm/test_cesm1_1_2/archive/b.e11.B1850C5CN.f19_g16.008.fwclim/ocn/hist/b.e11.B1850C5CN.f19_g16.008.fwclim.pop.h.1311-01.nc'
fout=fin[:-2]+'SDEN_F.nc'

ht_vars = ['SHF', 'SHF_QSW', 'LWDN_F', 'LWUP_F', 'SENH_F', 'MELTH_F', 'QFLUX']
fw_vars = ['SFWF', 'EVAP_F', 'PREC_F', 'ROFF_F',  'MELT_F', 'SALT_F', 'SNOW_F']

#Load SST/SSS
ds = xr.open_dataset(fin)
ds222 = xr.open_dataset(fin222)
non_dim_coords_reset=set(ds.coords)-set(ds.dims)
ds = ds.reset_coords(non_dim_coords_reset)

#IOFFFF = ds['IOFF_F']
#print(IOFFFF)

time=ds['time']
time.encoding['_FillValue']=None        # because xarray is weird
tlon = ds['TLONG']
tlat = ds['TLAT']
tarea = ds['TAREA']
ssd = ds['PD'].isel(z_t=0)
ssd=ssd.drop(['z_t'])
sst = ds['TEMP'].isel(z_t=0)
#sst = ds222['SST']
sss_psu = ds['SALT'].isel(z_t=0)
#sss_psu = ds222['SSS']
sss_msu=sss_psu.copy()/1000.
sss_msu.attrs['units']='kg/kg'
nt = np.shape(sst)[0]
ny = np.shape(sst)[1]
nx = np.shape(sst)[2]

# Compute rho, alpha, & beta using pop-tools.eos
depth=xr.DataArray(np.zeros(np.shape(sst)),dims=sst.dims,coords=sst.coords)
rho,drhods,drhodt = pop_tools.eos(salt=sss_psu,temp=sst,return_coefs=True,depth=depth)
alpha = (drhodt/rho)*-1		# 1/degC
beta = (drhods/rho)		# kg(Seawater)/kg(SALT)
sigma0 = rho - 1000.
sigma0=sigma0.drop(['z_t'])
sigma0.attrs['units']='kg/m^3'
sigma0.attrs['long_name']='Sigma referenced to z=0'
sigma0.encoding['_FillValue']=1.e30

depth=depth+1000.
tmp = pop_tools.eos(salt=sss_psu,temp=sst,return_coefs=False,depth=depth)
sigma1 = tmp - 1000.
sigma1=sigma1.drop(['z_t'])
sigma1.attrs['units']='kg/m^3'
sigma1.attrs['long_name']='Sigma referenced to z=1000'
sigma1.encoding['_FillValue']=1.e30

## check against alpha using gsw functions:
## first, convert model depth to pressure
##    (gsw function require numpy arrays, not xarray, so use ".values") 
##    (use [:,None,None] syntax to conform the z_t and tlat into 3D arrays)
#z_t = ds['z_t']
#temp = ds['TEMP']
#salt = ds['SALT']
#press = gsw.p_from_z(-z_t.values[:,None,None],tlat.values[None,:,:])
## compute absolute salinity from practical salinity
#SA = gsw.SA_from_SP(salt,press[None,:,:,:],tlon.values[None,None,:,:],tlat.values[None,None,:,:])
## compute conservative temperature from potential temperature
#CT = gsw.CT_from_pt(SA,temp.values)
#work = gsw.density.alpha(SA,CT,press)
#alpha_gsw = xr.DataArray(work[:,0,:,:],dims=['time','nlat','nlon'], \
#      coords={'time':time,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
#      name='ALPHA_gsw',attrs={'units':'1/K'})

#Load fluxes
ds_ht = ds222[ht_vars]
ds_fw = ds222[fw_vars]
#print(ds_fw)
#ht_vars = ['SHF', 'SHF_QSW', 'LWDN_F', 'LWUP_F', 'SENH_F', 'MELTH_F', 'QFLUX']
#fw_vars = ['SFWF', 'EVAP_F', 'PREC_F', 'ROFF_F',  'MELT_F', 'SALT_F', 'SNOW_F']
ds_ht['SHF'] = ds_ht['SHF'].mean(axis=0)
#print(ds_ht['SHF'].shape)
ds_ht['SHF_QSW'] = ds_ht['SHF_QSW'].mean(axis=0)
#print(ds_ht['SHF_QSW'].shape)

ds_ht['LWDN_F'] = ds_ht['LWDN_F'].mean(axis=0)
#print(ds_ht['LWDN_F'].shape)
ds_ht['LWUP_F'] = ds_ht['LWUP_F'].mean(axis=0)
#print(ds_ht['LWUP_F'].shape)

ds_ht['SENH_F'] = ds_ht['SENH_F'].mean(axis=0)
#print(ds_ht['SENH_F'].shape)
ds_ht['MELTH_F'] = ds_ht['MELTH_F'].mean(axis=0)
#print(ds_ht['MELTH_F'].shape)


ds_ht['QFLUX'] = ds_ht['QFLUX'].mean(axis=0)
#print(ds_ht['QFLUX'].shape)

ds_fw['SFWF'] = ds_fw['SFWF'].mean(axis=0)
#print(ds_fw['SFWF'].shape)
ds_fw['EVAP_F'] = ds_fw['EVAP_F'].mean(axis=0)
#print(ds_fw['EVAP_F'].shape)

ds_fw['PREC_F'] = ds_fw['PREC_F'].mean(axis=0)
#print(ds_fw['PREC_F'].shape)

ds_fw['ROFF_F'] = ds_fw['ROFF_F'].mean(axis=0)
#print(ds_fw['ROFF_F'].shape)
ds_fw['MELT_F'] = ds_fw['MELT_F'].mean(axis=0)
#print(ds_fw['MELT_F'].shape)

ds_fw['SALT_F'] = ds_fw['SALT_F'].mean(axis=0)
#print(ds_fw['SALT_F'].shape)

ds_fw['SNOW_F'] = ds_fw['SNOW_F'].mean(axis=0)
#print(ds_fw['SNOW_F'].shape)



#conversion factors
cpsw = ds222['cp_sw']                    # erg/g/K
cpsw = cpsw/1.e4
cpsw.attrs['units'] = 'J/Kg/K'
latvap=ds222['latent_heat_vapor']       # J/kg
latfus=ds222['latent_heat_fusion']      # erg/g
latfus = latfus/1.e4
latfus.attrs['units'] = 'J/kg'
sfluxfact = ds222['sflux_factor']        # (msu*cm/s)/(kg SALT /m^2/s)
salinityfact = ds222['salinity_factor']  # (msu*cm/s)/(kg FW /m^2/s)

#Perform needed heat flux manipulations
ds_ht['SHF'] = ds_ht['SHF'] + ds_ht['QFLUX']
#ds_ht['SHF'] = ds_ht['SHF'].mean(axis=0)
print(ds_ht['SHF'].shape)		# Net heat flux
ds_ht['LWDN_F'] = ds_ht['LWDN_F'] + ds_ht['LWUP_F']	# Net LW
#ds_ht['LWDN_F'] = ds_ht['LWDN_F'].mean(axis=0)
# Compute Latent heat flux
latent=ds_fw['EVAP_F']*latvap
#latent = latent.mean(axis=0)
latent.attrs = {'units':'W/m^2','long_name':'Latent Heat Flux'}
ds_ht['Latent']=latent
# Compute Snowmelt heat flux
snowmelt=ds_fw['SNOW_F']*latfus*-1
#snowmelt=snowmelt.mean(axis=0)
print(snowmelt.shape)
snowmelt.attrs = {'units':'W/m^2','long_name':'Heat Flux due to snow melt'}
ds_ht['Snowmelt']=snowmelt
# Compute heat flux due to solid river runoff
ioffmelt=0*latfus*-1
#ioffmelt = ioffmelt.mean(axis=0)
#iprint('p',ioffmelt.shape)
ioffmelt.attrs = {'units':'W/m^2','long_name':'Heat Flux due to solid runoff melt'}
ds_ht['IOFFmelt']=ioffmelt
# rename Q components
qcomps = ['Total', 'Shortwave', 'Longwave', 'Latent', 'Sensible', 'Icemelt', 'Snowmelt', 'IOFFmelt', 'Frazil']
ds_ht=ds_ht.rename({'SHF':'Total'})
ds_ht=ds_ht.rename({'SHF_QSW':'Shortwave'})
ds_ht=ds_ht.rename({'LWDN_F':'Longwave'})
ds_ht=ds_ht.rename({'SENH_F':'Sensible'})
ds_ht=ds_ht.rename({'MELTH_F':'Icemelt'})
ds_ht=ds_ht.rename({'QFLUX':'Frazil'})

#Perform needed freshwater flux manipulations
ds_fw['SFWF'] = ds_fw['SFWF'] - ds_ht['Frazil']/latfus
#ds_fw['SFWF'] = ds_fw['SFWF'].mean(axis=0)
print(ds_fw['SFWF'].shape)	
ds_fw['SALT_F'] = ds_fw['SALT_F']*sfluxfact/salinityfact	# (kg salt/m^2/s to kg freshwater/m^2/s); units remain unchanged
#ds_fw['SALT_F'] = ds_fw['SALT_F'].mean(axis=0)

qflux = ds_ht['Frazil']/latfus*-1
#qflux = qflux.mean(axis=0)
qflux.attrs = {'units':'kg/m^2/s','long_name':'Freshwater Flux due to Qflux'}
ds_fw['Frazil'] = qflux
# rename FW components
ds_fw=ds_fw.rename({'SFWF':'Total'})
ds_fw=ds_fw.rename({'PREC_F':'Precip'})
ds_fw=ds_fw.rename({'EVAP_F':'Evap'})
ds_fw=ds_fw.rename({'ROFF_F':'ROFF'})
#ds_fw=ds_fw.rename({'IOFF_F':'IOFF'})
ds_fw=ds_fw.rename({'MELT_F':'Icemelt'})
ds_fw=ds_fw.rename({'SALT_F':'BrineReject'})

#========================================================================================
# Compute Surface Density Flux
#
# References
# ----------
# Speer & Tziperman, 1992: Rates of Water Mass Formation in the North 
# Atlantic Ocean, J. Phys. Oceanogr., 22, 93-104.
#
# Large, W. G. and G. Nurser, 2001: Chapter 5.1 ocean surface water mass transformation. Ocean
# Circulation and Climate – Observing and Modelling the Global Ocean, Academic Press, International
# Geophysics, Vol. 77, 317–336, doi:10.1016/S0074-6142(01)80126-1.
#
#	SDENS_F = -(alpha*SHF)/C_p    + (beta*SFWF)*(SSS/(1-SSS))
#
# NOTE on units:
# 
#    mass FW = mass SW - mass SALT
#            = rho*dV*(1 - S), since S == (mass SALT)/(mass SW)
#    ==> (1-S) == (mass FW)/(mass SW)
# 
#    F == (mass FW)/(m^2s), so F*S/(1-S) == [(kg FW)/(m^2s)]*[(kg SALT)/(kg SW)]
#                                           ------------------------------------
#                                                      (kg FW)/(kg SW)
#    Beta = (1/psu) = (kg SW)/(kg SALT), so
#    Beta * F * S/(1-S) == (kg SW)/(m^2 s) is a mass flux of SW
# 
# Origin of "rho_o" in some publications is that it is needed to convert F from
# cm/year or whateever into kg/m^2/s.
#----------------------------------------------------------------------------------------

# set up output xarrays
qcomps = ['Total', 'Shortwave', 'Longwave', 'Latent', 'Sensible', 'Icemelt', 'Snowmelt', 'IOFFmelt', 'Frazil']
ncompq = np.size(qcomps)
fcomps = ['Total', 'Precip','Evap','ROFF','Icemelt','BrineReject','Frazil']
ncompf = np.size(fcomps)
SDEN = xr.DataArray(np.zeros((nt,ny,nx),dtype=np.float32),dims=['time','nlat','nlon'], \
      coords={'time':time.data,'TLAT':(('nlat','nlon'),tlat.data),'TLONG':(('nlat','nlon'),tlon.data)}, \
      name='SDEN',attrs={'long_name':'Surface Density Flux','units':'kg/m^2/s'})
SDEN.encoding['_FillValue']=1.e30
SDEN_Q = xr.DataArray(np.zeros((nt,ncompq,ny,nx),dtype=np.single),dims=['time','qcomp','nlat','nlon'], \
      coords={'time':time.data,'qcomp':qcomps,'TLAT':(('nlat','nlon'),tlat.data),'TLONG':(('nlat','nlon'),tlon.data)}, \
      name='SDEN_Q',attrs={'long_name':'Surface Density Flux (Heat)','units':'kg/m^2/s'})
SDEN_Q.encoding['_FillValue']=1.e30
SDEN_F = xr.DataArray(np.zeros((nt,ncompf,ny,nx),dtype=np.single),dims=['time','fcomp','nlat','nlon'], \
      coords={'time':time.data,'fcomp':fcomps,'TLAT':(('nlat','nlon'),tlat.data),'TLONG':(('nlat','nlon'),tlon.data)}, \
      name='SDEN_F',attrs={'long_name':'Surface Density Flux (Freshwater)','units':'kg/m^2/s'})
SDEN_F.encoding['_FillValue']=1.e30

# Compute surface density flux and fill xarrays
#ds_ht['Total']=ds_ht['Total'].mean(axis=0)
#ds_fw['Total'] = ds_fw['Total'].mean(axis=0)
work = (-alpha*ds_ht['Total']/cpsw - beta*ds_fw['Total']*(sss_msu/(1-sss_msu))).astype(np.float32)


#print(ds_ht['Total'].shape)

#print(ds_fw['Total'].shape)
print(work.shape,'work')
#print(SDEN.shape)

SDEN.values[:,:]=work.values[:,:]

# Compute surface density flux due to heat flux and add to the new dataset
for i in qcomps:
   work = (-alpha*ds_ht[i]/cpsw).astype(np.float32)
   
   j = qcomps.index(i)
   SDEN_Q.values[:,j,:,:] = work.values

# Compute surface density flux due to freshwater flux and add to the new dataset
for i in fcomps:
   work = (-beta*ds_fw[i]*(sss_msu/(1-sss_msu))).astype(np.float32)
   j = fcomps.index(i)
   SDEN_F.values[:,j,:,:] = work.values

# Write SDF dataset in netcdf
out_ds=SDEN.to_dataset(name='SDEN_F')
out_ds.TLONG.attrs=tlon.attrs
out_ds.TLONG.encoding['_FillValue']=None        # because xarray is weird
out_ds.TLAT.attrs=tlat.attrs
out_ds.TLAT.encoding['_FillValue']=None        # because xarray is weird
out_ds['SDEN_F_Q']=SDEN_Q
out_ds['SDEN_F_F']=SDEN_F
#out_ds['alpha']=alpha
#out_ds['alpha_gsw']=alpha_gsw
#out_ds['beta']=beta
out_ds['sigma0']=sigma0
out_ds['sigma1']=sigma1
out_ds['TAREA']=tarea
#out_ds['SSD']=ssd
out_ds.to_netcdf(fout,unlimited_dims='time')

time2=timer.time()
print('DONE creating ',fout,'. Total time = ',time2-time1,'s')
