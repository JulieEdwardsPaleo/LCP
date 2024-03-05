

from matplotlib.ticker import MultipleLocator
import numpy as np  
import xarray as xr
import pandas as pd
import os
import fnmatch
import scipy as sp
from scipy.stats import pearsonr
from pca import pca
import numpy as np
from matplotlib import pyplot as plt
import utils as u
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Rectangle

## DATA READ IN 

## Climate data
#temperature - CRU
Workdir='/Users/julieedwards/Documents/Projects/LCP/climate_data/cru/'
os.chdir(Workdir)
#Tmean
ncfile = 'icru4_tmp_-105.22E_36.83N_n.nc'
with xr.open_dataarray(ncfile,decode_times=False).load() as Tmean_CRU:
    units, reference_date = Tmean_CRU.time.attrs['units'].split('since')
    Tmean_CRU['time'] = pd.date_range(start=reference_date, periods=Tmean_CRU.sizes['time'], freq='M')
    Year = Tmean_CRU['time'].dt.strftime('%Y').values
    Month = Tmean_CRU['time'].dt.strftime('%m').values
# TMax
ncfile = 'icru4_tmx_-105.22E_36.83N_n.nc'
with xr.open_dataarray(ncfile,decode_times=False).load() as Tmax_CRU:
    units, reference_date = Tmax_CRU.time.attrs['units'].split('since')
    Tmax_CRU['time'] = pd.date_range(start=reference_date, periods=Tmax_CRU.sizes['time'], freq='M')
#precipitation - CRU
ncfile = 'icru4_pre_-105.22E_36.83N_n.nc'
with xr.open_dataarray(ncfile,decode_times=False).load() as Precip_CRU:
    units, reference_date = Precip_CRU.time.attrs['units'].split('since')
    Precip_CRU['time'] = pd.date_range(start=reference_date, periods=Precip_CRU.sizes['time'], freq='M')
#SPEI - CSIC SPEI
ncfile = 'ispei_01_-105.22E_36.83N_n.nc'
with xr.open_dataarray(ncfile,decode_times=False).load() as Spei_CRU:
    units, reference_date = Spei_CRU.time.attrs['units'].split('since')
    Spei_CRU['time'] = pd.date_range(start=reference_date, periods=Spei_CRU.sizes['time'], freq='M')
    Spei_Year=Spei_CRU['time'].dt.strftime('%Y').values
    Spei_Month = Spei_CRU['time'].dt.strftime('%m').values

##VPD data

    #TopoWx Temperature data
Workdir='/Users/julieedwards/Documents/Projects/LCP/climate_data/'
os.chdir(Workdir)

topo_avg_file='LCH_topo_tmean.csv'
topo_avg=pd.read_csv(topo_avg_file,index_col=None,header=None)
topoMH=pd.read_csv('LCHtopoMonthly.csv',index_col=None,header=None)
topoML=pd.read_csv('LCLtopoMonthly.csv',index_col=None,header=None)
WaterYear=pd.read_csv('TotalWaterYearPrecip.csv',index_col=None,header=None)

## QWA Data - 30 year spline and linear detrend 
Series= ["MXCWTRAD","MXDCWA","EWW","MRW","LWW","MXCWTTAN","MLA.EW"]
reso="_20mu"
method="q75"

###############HIGH ELEVATION
Workdir='/Users/julieedwards/Documents/Projects/LCP/QWA_High/concat/Summary/'
os.chdir(Workdir)
files = os.listdir(Workdir)
############### 30 year spline chronologies
relevant_files20 = []
for file_name in files:
    for i in range(len(Series)):
        if fnmatch.fnmatch(file_name, f"Spline30*{method}*{reso}*"):
            relevant_files20.append(file_name)
print(set(relevant_files20))
relevant_files20=set(relevant_files20)
li=[]
df=[]
for filename in relevant_files20:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df["std"])
frame = pd.concat(li, axis=1, ignore_index=True)
frame.columns = relevant_files20

uHiSpline20=frame.set_index(df["Unnamed: 0"]) #20um spline################################

MRWseries=pd.read_csv('ARSTAN_tucson_MRW_q75_20mu_gap.txt',index_col=0,header=0)
MRWseries=MRWseries[MRWseries.index>1916]
EWLAseries=pd.read_csv('ARSTAN_tucson_MLA.EW_q75_20mu_gap.txt',index_col=0,header=0)
EWLAseries=EWLAseries[EWLAseries.index>1916]

###########################################################################

###############LOW ELEVATION

Workdir='/Users/julieedwards/Documents/Projects/LCP/QWA_Low/concat/Summary/'
os.chdir(Workdir)
files = os.listdir(Workdir)
reso="_20mu"
############### 30 year spline chronologies
relevant_files20 = []
for file_name in files:
    for i in range(len(Series)):
        if fnmatch.fnmatch(file_name, f"Spline30*{method}*{reso}*"):
            relevant_files20.append(file_name)
print(set(relevant_files20))
relevant_files20=set(relevant_files20)
li=[]
df=[]
for filename in relevant_files20:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df["std"])
frame = pd.concat(li, axis=1, ignore_index=True)
frame.columns = relevant_files20
frame.set_index(df["Unnamed: 0"])

uLoSpline20=frame.set_index(df["Unnamed: 0"])


# Figure 1 climatology plot#####################################
Tmean=pd.DataFrame(Tmean_CRU)
Tmean['Month']=Month.astype(np.int64)
Tmean['Year']=Year.astype(np.int64)
#exsitu=Tmean[Tmean['Year']==2019][0]
#exsitu.index=range(1,13)
Premean=pd.DataFrame(Precip_CRU)
Premean['Month']=Month.astype(np.int64)
Premean['Year']=Year.astype(np.int64)
#Pexsitu=Premean[Premean['Year']==2019][0]
#Pexsitu.index=range(1,13)

T_climatology=Tmean_CRU.groupby("time.month").mean("time")
T_std=Tmean_CRU.groupby("time.month").std("time")

P_climatology=Precip_CRU.groupby("time.month").mean("time")
P_std=Precip_CRU.groupby("time.month").std("time")

fig, ax1 = plt.subplots()
plt.title('Little Costilla Peak climate')
color = [0.8392,0.3765,0.3020]
ax1.set_xlabel('Month')
ax1.set_ylabel('Temperature ($^\circ$C)', color=color)
ax1.plot(T_climatology["month"], T_climatology, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(-15,50)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.set_xlim(1,12)
ax1.fill_between(T_climatology["month"], T_climatology-T_std, T_climatology+T_std,alpha=0.2,color=color)
ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
H=ax1.plot(T_climatology["month"],topoMH,color='black',ls='--',label='LCH TopoWx (1948-2016)')
L=ax1.plot(T_climatology["month"],topoML,color='black',ls=':',label='LCL TopoWx (1948-2016)')
leg=plt.legend(loc='upper left',fontsize=8)
leg.get_frame().set_edgecolor('w')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = [0.1294,0.4000,0.6745]
ax2.set_ylabel('Precipitation (mm)', color=color)  # we already handled the x-label with ax1
ax2.plot(P_climatology["month"], P_climatology, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(-30,100)
ax2.set_xlim(1,12)
ax2.fill_between(T_climatology["month"], P_climatology-P_std, P_climatology+P_std,alpha=0.2,color=color)
ax2.spines[['right', 'top']].set_visible(True)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/climatology.eps', format="eps",transparent=True)
plt.close()


################### PCA######################
Hdf=uHiSpline20
Hdf=Hdf[Hdf.index>1916]
SeriesOrder=["Spline30yr_tucson_MRW_q75_20mu_gap.txt","Spline30yr_tucson_EWW_q75_20mu_gap.txt","Spline30yr_tucson_MXCWTRAD_q75_20mu_gap.txt","Spline30yr_tucson_MXDCWA_q75_20mu_gap.txt","Spline30yr_tucson_MXCWTTAN_q75_20mu_gap.txt","Spline30yr_tucson_LWW_q75_20mu_gap.txt","Spline30yr_tucson_MLA.EW_q75_20mu_gap.txt"]
currentOrder=Hdf.columns.values
Hdf = Hdf[SeriesOrder]

scaling=StandardScaler()
# Use fit and transform method 
scaling.fit(Hdf[Hdf.index>1916])
Scaled_data=scaling.transform(Hdf)
C = np.corrcoef(Scaled_data,rowvar=False,ddof=1)
# since this is a reasonably sized dataset, we can get eigenvectors and eigenvalues from np.linalg.svd
U,S,V = np.linalg.svd(C)
A = U[:,0:2]
model = pca(normalize=True, n_components=2)
results=model.fit_transform(Hdf)
exvar=results['explained_var']
#################PLOTTING
colors=[(0,0,0),(0.8392,0.3765,0.3020),(0.2627,0.5765,0.7647),(0.2627,0.5765,0.7647),(0.2627,0.5765,0.7647),(0.2627,0.5765,0.7647),(0.8392,0.3765,0.3020)]
SeriesLabel=["TRW","EWW","MxRCWT","aMXD","MxTCWT","LWW","EWLA"]

fig, axs = plt.subplots(2, 2,figsize=(8, 7))
for i in range(len(Series)):
    axs[0,1].plot([0,U[i,0]*-1],[0,U[i,1]*-1],markersize=10,color=colors[i])
    axs[0,1].plot(U[i,0]*-1,U[i,1]*-1,'o',markersize=10,color=colors[i])
axs[0,0].set_xlim(-.8,.8)
axs[0,0].set_ylim(-.8,.8)
axs[0,0].set_xticks(np.linspace(-.8,.8,9))
axs[0,0].set_yticks(np.linspace(-.8,.8,9))
axs[0,0].spines[['right', 'top']].set_visible(True)
axs[0,0].hlines(0,-1,1,'k')
axs[0,0].vlines(0,-1,1,'k')
axs[0,1].set_ylabel(f'PC2: {(exvar[1]-exvar[0]) * 100:.1f}%')
axs[0,1].set_xlabel(f'PC1: {exvar[0] * 100:.1f}%')
axs[0,1].text(-.8, .85, 'b)')
for i in range(len(Series)):
    axs[0,1].text(U[i,0]*-1,U[i,1]*-1,SeriesLabel[i])
axs[0,1].set_title('LCH QWA PCA',fontsize=12,y=1)
axs[0,0].grid(color=(.5,.5,.5), linestyle='--', linewidth=.5, alpha=0.3)
latewood=axs[0,0].plot([], [], 'o',c=(0.2627,0.5765,0.7647), markersize=10,label='LW')
earlywood=axs[0,0].plot([], [], 'o',c=(0.8392,0.3765,0.3020), markersize=10,label='EW')
full=axs[0,0].plot([], [], 'o',c='k', markersize=10,label='TRW')
leg=axs[0,0].legend(loc='lower left')
leg.get_frame().set_edgecolor('w')
#plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/pca_high.eps', format='eps')
#plt.close()


Ldf=uLoSpline20
Ldf=Ldf[Ldf.index>1916]
SeriesOrder=["Spline30yr_tucson_MRW_q75_20mu_gap.txt","Spline30yr_tucson_EWW_q75_20mu_gap.txt","Spline30yr_tucson_MXCWTRAD_q75_20mu_gap.txt","Spline30yr_tucson_MXDCWA_q75_20mu_gap.txt","Spline30yr_tucson_MXCWTTAN_q75_20mu_gap.txt","Spline30yr_tucson_LWW_q75_20mu_gap.txt","Spline30yr_tucson_MLA.EW_q75_20mu_gap.txt"]
currentOrder=Ldf.columns.values
Ldf = Ldf[SeriesOrder]
scaling=StandardScaler()
scaling.fit(Ldf[Ldf.index>1916])
Scaled_data=scaling.transform(Ldf)
C = np.corrcoef(Scaled_data,rowvar=False,ddof=1)
# since this is a reasonably sized dataset, we can get eigenvectors and eigenvalues from np.linalg.svd
U,S,V = np.linalg.svd(C)
A = U[:,0:2] 
model = pca(normalize=True, n_components=2)
results=model.fit_transform(Ldf)
exvar=results['explained_var']
#################PLOTTING
for i in range(len(Series)):
    axs[0,0].plot([0,U[i,0]*-1],[0,U[i,1]*-1],markersize=10,color=colors[i])
    axs[0,0].plot(U[i,0]*-1,U[i,1]*-1,'o',markersize=10,color=colors[i])
axs[0,1].set_xlim(-.8,.8)
axs[0,1].set_ylim(-.8,.8)
axs[0,1].set_xticks(np.linspace(-.8,.8,9))
axs[0,1].set_yticks(np.linspace(-.8,.8,9))
axs[0,1].spines[['right', 'top']].set_visible(True)
axs[0,1].hlines(0,-1,1,'k')
axs[0,1].vlines(0,-1,1,'k')
axs[0,0].set_ylabel(f'PC2: {(exvar[1]-exvar[0]) * 100:.1f}%')
axs[0,0].set_xlabel(f'PC1: {exvar[0] * 100:.1f}%')
axs[0,0].text(-.8, .85, 'a)')
for i in range(len(Series)):
    axs[0,0].text(U[i,0]*-1,U[i,1]*-1,SeriesLabel[i])
axs[0,0].set_title('LCL QWA PCA',fontsize=12,y=1)
axs[0,1].grid(color=(.5,.5,.5), linestyle='--', linewidth=.5, alpha=0.3)
latewood=axs[0,1].plot([], [], 'o',c=(0.2627,0.5765,0.7647), markersize=10,label='LW')
earlywood=axs[0,1].plot([], [], 'o',c=(0.8392,0.3765,0.3020), markersize=10,label='EW')
full=axs[0,1].plot([], [], 'o',c='k', markersize=10,label='TRW')
leg=axs[0,1].legend(loc='lower left')
leg.get_frame().set_edgecolor('w')
#plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/pca_t.eps', format='eps')
#plt.close()
##### PCA continued + rotated ########
scaling=StandardScaler()
 
# Use fit and transform method 
scaling.fit(Hdf[Hdf.index>1916])
Scaled_data=scaling.transform(Hdf)
# Set the n_components=
principal=PCA(n_components=2)
principal.fit(Scaled_data)
Hx=principal.transform(Scaled_data)

HiScore=pd.DataFrame(Hx*-1)
HiScore.index=Hdf.index
HiScore.columns=['PC1','PC2']

#rotation

# get the correlation matrix
C = np.corrcoef(Scaled_data,rowvar=False,ddof=1)
# since this is a reasonably sized dataset, we can get eigenvectors and eigenvalues from np.linalg.svd
U,S,V = np.linalg.svd(C)
A = U[:,0:2]
Hpcs = Hdf @ A
Hpcs.columns=['PC1','PC2']

Hreofs, rotation_matrix = u.varimax_rotation(A,normalize=True)
Hrpcs = pd.DataFrame(Scaled_data @ Hreofs)
Hrpcs.index=Hdf.index
Hrpcs.columns=['RPC1','RPC2']
np.corrcoef(Hrpcs,rowvar=False)
Hexvar=S/np.trace(C)


SeriesLabel=["TRW","EWW","MxRCWT","aMXD","MxTCWT","LWW","EWLA"]


scaling.fit(Ldf[Ldf.index>1916])
Scaled_data=scaling.transform(Ldf)

# Set the n_components=3
principal=PCA(n_components=2)
principal.fit(Scaled_data)
Lx=principal.transform(Scaled_data)

LoScore=pd.DataFrame(Lx*-1)
LoScore.index=Ldf.index
LoScore.columns=['PC1','PC2']
loadings = pd.DataFrame(principal.components_.T*-1, columns=['PC1', 'PC2'], index=SeriesLabel)


#rotation

scaling.fit(Ldf[Ldf.index>1916])
Scaled_data=scaling.transform(Ldf)
C = np.corrcoef(Scaled_data,rowvar=False,ddof=1)
# since this is a reasonably sized dataset, we can get eigenvectors and eigenvalues from np.linalg.svd
U,S,V = np.linalg.svd(C)

A = U[:,0:2] 
Lpcs = Ldf @ A
Lpcs.columns=['PC1','PC2']

Lreofs, rotation_matrix = u.varimax_rotation(A,normalize=True)
Lrpcs = pd.DataFrame(Scaled_data @ Lreofs)
Lrpcs.index=Ldf.index
Lrpcs.columns=['RPC1','RPC2']
np.corrcoef(Lrpcs,rowvar=False)
Lexvar=S/np.trace(C)



######

for i in range(len(Series)):
    axs[1,1].plot([0,Hreofs[i,0]*-1],[0,Hreofs[i,1]*-1],markersize=10,color=colors[i])
    axs[1,1].plot(Hreofs[i,0]*-1,Hreofs[i,1]*-1,'o',markersize=10,color=colors[i])
axs[1,0].set_xlim(-.8,.8)
axs[1,0].set_ylim(-.8,.8)
axs[1,0].set_xticks(np.linspace(-.8,.8,9))
axs[1,0].set_yticks(np.linspace(-.8,.8,9))
axs[1,0].spines[['right', 'top']].set_visible(True)
axs[1,0].hlines(0,-1,1,'k')
axs[1,0].vlines(0,-1,1,'k')
axs[1,0].set_ylabel('RPC2')
axs[1,0].set_xlabel('RPC1')
axs[1,0].text(-.8, .85, 'c)')
for i in range(len(Series)):
    axs[1,1].text(Hreofs[i,0]*-1,Hreofs[i,1]*-1,SeriesLabel[i])
axs[1,1].set_title('LCH QWA Rotated PCA',fontsize=12,y=1)
axs[1,0].grid(color=(.5,.5,.5), linestyle='--', linewidth=.5, alpha=0.3)
latewood=axs[1,0].plot([], [], 'o',c=(0.2627,0.5765,0.7647), markersize=10,label='LW')
earlywood=axs[1,0].plot([], [], 'o',c=(0.8392,0.3765,0.3020), markersize=10,label='EW')
full=axs[1,0].plot([], [], 'o',c='k', markersize=10,label='TRW')
leg=axs[1,0].legend(loc='lower left')
leg.get_frame().set_edgecolor('w')
#plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/Rpca_high.eps', format='eps')
#plt.close()

for i in range(len(Series)):
    axs[1,0].plot([0,Lreofs[i,0]*-1],[0,Lreofs[i,1]*-1],markersize=10,color=colors[i])
    axs[1,0].plot(Lreofs[i,0]*-1,Lreofs[i,1]*-1,'o',markersize=10,color=colors[i])
axs[1,1].set_xlim(-.8,.8)
axs[1,1].set_ylim(-.8,.8)
axs[1,1].set_xticks(np.linspace(-.8,.8,9))
axs[1,1].set_yticks(np.linspace(-.8,.8,9))
axs[1,1].spines[['right', 'top']].set_visible(True)
axs[1,1].hlines(0,-1,1,'k')
axs[1,1].vlines(0,-1,1,'k')
axs[1,1].set_ylabel('RPC2')
axs[1,1].set_xlabel('RPC1')
axs[1,1].text(-.8, .85, 'd)')
for i in range(len(Series)):
    axs[1,0].text(Lreofs[i,0]*-1,Lreofs[i,1]*-1,SeriesLabel[i])
axs[1,0].set_title('LCL QWA Rotated PCA',fontsize=12,y=1)
axs[1,1].grid(color=(.5,.5,.5), linestyle='--', linewidth=.5, alpha=0.3)
latewood=axs[1,1].plot([], [], 'o',c=(0.2627,0.5765,0.7647), markersize=10,label='LW')
earlywood=axs[1,1].plot([], [], 'o',c=(0.8392,0.3765,0.3020), markersize=10,label='EW')
full=axs[1,1].plot([], [], 'o',c='k', markersize=10,label='TRW')
leg=axs[1,1].legend(loc='lower left')
leg.get_frame().set_edgecolor('w')
fig.tight_layout(pad=1.5)

plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/PCA_ALL.eps', format='eps')
plt.close()


##############################
####### Climate correlations ###########################
 
##############################
####### Climate correlations ###########################
Tmax=pd.DataFrame(Tmax_CRU)
Tmax['Month']=Month.astype(np.int64)
Tmax['Year']=Year.astype(np.int64)
Precip=pd.DataFrame(Precip_CRU)
Precip['Month']=Month.astype(np.int64)
Precip['Year']=Year.astype(np.int64)
Spei=pd.DataFrame(Spei_CRU)
Spei['Month']=Spei_Month.astype(np.int64)
Spei['Year']=Spei_Year.astype(np.int64)

StartYear=1917
EndYear=2017

Tmax=Tmax[(Tmax['Year']>=StartYear) & (Tmax['Year']<=EndYear)]
Precip=Precip[(Precip['Year']>=StartYear) & (Precip['Year']<=EndYear)]
Spei=Spei[(Spei['Year']>=StartYear) & (Spei['Year']<=EndYear)]

SeriesOrder=['RPC2', 'PC2','Spline30yr_tucson_MLA.EW_q75_20mu_gap.txt','Spline30yr_tucson_EWW_q75_20mu_gap.txt','Spline30yr_tucson_MRW_q75_20mu_gap.txt', 'RPC1','PC1','Spline30yr_tucson_MXDCWA_q75_20mu_gap.txt','Spline30yr_tucson_MXCWTRAD_q75_20mu_gap.txt','Spline30yr_tucson_MXCWTTAN_q75_20mu_gap.txt','Spline30yr_tucson_LWW_q75_20mu_gap.txt']
#SeriesOrder=['RPC2','RPC1','PC2','PC1',"Spline30yr_tucson_MXDCWA_q75_20mu_gap.txt","Spline30yr_tucson_MXCWTRAD_q75_20mu_gap.txt","Spline30yr_tucson_MXCWTTAN_q75_20mu_gap.txt","Spline30yr_tucson_LWW_q75_20mu_gap.txt","Spline30yr_tucson_MLA.EW_q75_20mu_gap.txt","Spline30yr_tucson_EWW_q75_20mu_gap.txt","Spline30yr_tucson_MRW_q75_20mu_gap.txt"]
HiSpline20=pd.concat([Hdf,Hpcs*-1,Hrpcs*-1],axis=1)
currentOrder=HiSpline20.columns.values
HiSpline20 = HiSpline20[SeriesOrder]

period=30#Spline 50% freq cutoff
temp_r=[]
temp_p=[]
precip_r=[]
precip_p=[]
spei_r=[]
spei_p=[]
for i in range(0,12):
    for j in range(len(Series)+4):
        Tmonthdetr = u.spline(Tmax[Tmax['Month']==i+1]['Year'], Tmax[Tmax['Month']==i+1][0], period)
        r_temp,p_temp=pearsonr(Tmonthdetr,HiSpline20[HiSpline20.index>=StartYear][HiSpline20.columns[j]], alternative='two-sided')
        temp_r.append(r_temp)
        temp_p.append(p_temp)
        Pmonthdetr = u.spline(Precip[Precip['Month']==i+1]['Year'], Precip[Precip['Month']==i+1][0], period)
        r_precip,p_precip=pearsonr(Pmonthdetr,HiSpline20[HiSpline20.index>=StartYear][HiSpline20.columns[j]], alternative='two-sided')
        precip_r.append(r_precip)
        precip_p.append(p_precip)
        Smonthdetr = u.spline(Spei[Spei['Month']==i+1]['Year'], Spei[Spei['Month']==i+1][0], period)
        r_spei,p_spei=pearsonr(Smonthdetr,HiSpline20[HiSpline20.index>=StartYear][HiSpline20.columns[j]], alternative='two-sided')
        spei_r.append(r_spei)
        spei_p.append(p_spei)        

Tr = np.array(temp_r).reshape(12,len(Series)+4).transpose()
Pr = np.array(precip_r).reshape(12,len(Series)+4).transpose()
Sr = np.array(spei_r).reshape(12,len(Series)+4).transpose()
Tp = np.array(temp_p).reshape(12,len(Series)+4).transpose()
Pp = np.array(precip_p).reshape(12,len(Series)+4).transpose()
Sp = np.array(spei_p).reshape(12,len(Series)+4).transpose()



############### PLOTTING
SeriesLabel=['RPC2', 'PC2','EWLA','EWW','TRW', 'RPC1','PC1','aMXD','MxRCWT','MxTCWT','LWW']
sig_temp=np.where(Tp < 0.01)
sig_precip=np.where(Pp < 0.01)
sig_spei=np.where(Sp<0.01)

months=["J","F","M","A","M","J","J","A","S","O","N","D"]

dx, dy = 1, 1
y, x = np.mgrid[0:10+dy:dy, 1:12+dx:dx]
z=Tr
zp=Pr
zs=Sr
fig, axs = plt.subplots(3, 2,figsize=(12, 8))
r=axs[0,1].pcolormesh(x,y,z, cmap='RdBu_r',clim=[-.6,.6])
pr=axs[1,1].pcolormesh(x,y,zp, cmap='RdBu_r',clim=[-.6,.6])
pr=axs[2,1].pcolormesh(x,y,zs, cmap='RdBu_r',clim=[-.6,.6])
#fig.suptitle('LCH',fontsize=20,y=0.96)
axs[0,1].spines[['right', 'top']].set_visible(True)
axs[1,1].spines[['right', 'top']].set_visible(True)
axs[2,1].spines[['right', 'top']].set_visible(True)

plt.xlabel('Month')
axs[0,1].scatter(x[sig_temp], y[sig_temp], marker = '.',color='k')
axs[1,1].scatter(x[sig_precip], y[sig_precip], marker = '.',color='k')
axs[2,1].scatter(x[sig_spei], y[sig_spei], marker = '.',color='k')
axs[0,1].set_title('Maximum Temperature')
axs[1,1].set_title('Precipitation')
axs[2,1].set_title('SPEI')
#fig.tight_layout(pad=1.5)
axs[0,1].xaxis.set_major_locator(MultipleLocator(1))
axs[1,1].xaxis.set_major_locator(MultipleLocator(1))
axs[2,1].xaxis.set_major_locator(MultipleLocator(1))
axs[0,1].yaxis.set_major_locator(MultipleLocator(1))
axs[1,1].yaxis.set_major_locator(MultipleLocator(1))
axs[2,1].yaxis.set_major_locator(MultipleLocator(1))
axs[0,0].set_yticks(range(11), labels=SeriesLabel)
axs[1,0].set_yticks(range(11), labels=SeriesLabel)
axs[2,0].set_yticks(range(11), labels=SeriesLabel)
axs[0,1].set_xticks(range(1,13), labels=months)
axs[1,1].set_xticks(range(1,13), labels=months)
axs[2,1].set_xticks(range(1,13), labels=months)



## LOW ELEVATON #############

#SeriesOrder=['RPC2','RPC1','PC2','PC1',"Spline30yr_tucson_MXDCWA_q75_20mu_gap.txt","Spline30yr_tucson_MXCWTRAD_q75_20mu_gap.txt","Spline30yr_tucson_MXCWTTAN_q75_20mu_gap.txt","Spline30yr_tucson_LWW_q75_20mu_gap.txt","Spline30yr_tucson_MLA.EW_q75_20mu_gap.txt","Spline30yr_tucson_EWW_q75_20mu_gap.txt","Spline30yr_tucson_MRW_q75_20mu_gap.txt"]
LoSpline20=pd.concat([Ldf,Lpcs*-1,Lrpcs*-1],axis=1)
currentOrder=LoSpline20.columns.values
LoSpline20 = LoSpline20[SeriesOrder]


period=30#Spline 50% freq cutoff
temp_r=[]
temp_p=[]
precip_r=[]
precip_p=[]
spei_r=[]
spei_p=[]
for i in range(0,12):
    for j in range(len(Series)+4):
        Tmonthdetr = u.spline(Tmax[Tmax['Month']==i+1]['Year'], Tmax[Tmax['Month']==i+1][0], period)
        r_temp,p_temp=pearsonr(Tmonthdetr,LoSpline20[LoSpline20.index>=StartYear][LoSpline20.columns[j]], alternative='two-sided')
        temp_r.append(r_temp)
        temp_p.append(p_temp)
        Pmonthdetr = u.spline(Precip[Precip['Month']==i+1]['Year'], Precip[Precip['Month']==i+1][0], period)
        r_precip,p_precip=pearsonr(Pmonthdetr,LoSpline20[LoSpline20.index>=StartYear][LoSpline20.columns[j]], alternative='two-sided')
        precip_r.append(r_precip)
        precip_p.append(p_precip)
        Smonthdetr = u.spline(Spei[Spei['Month']==i+1]['Year'], Spei[Spei['Month']==i+1][0], period)
        r_spei,p_spei=pearsonr(Smonthdetr,LoSpline20[LoSpline20.index>=StartYear][LoSpline20.columns[j]], alternative='two-sided')
        spei_r.append(r_spei)
        spei_p.append(p_spei)        

Tr = np.array(temp_r).reshape(12,len(Series)+4).transpose()
Pr = np.array(precip_r).reshape(12,len(Series)+4).transpose()
Sr = np.array(spei_r).reshape(12,len(Series)+4).transpose()
Tp = np.array(temp_p).reshape(12,len(Series)+4).transpose()
Pp = np.array(precip_p).reshape(12,len(Series)+4).transpose()
Sp = np.array(spei_p).reshape(12,len(Series)+4).transpose()



############### PLOTTING
#SeriesLabel=["RPC2","RPC1","PC2","PC1","aMXD","MxRCWT","MxTCWT","LWW","EWLA","EWW","TRW"]
sig_temp=np.where(Tp < 0.01)
sig_precip=np.where(Pp < 0.01)
sig_spei=np.where(Sp<0.01)

months=["J","F","M","A","M","J","J","A","S","O","N","D"]

dx, dy = 1, 1
y, x = np.mgrid[0:10+dy:dy, 1:12+dx:dx]
z=Tr
zp=Pr
zs=Sr
#fig, axs = plt.subplots(3, 2,figsize=(10, 10))
r=axs[0,0].pcolormesh(x,y,z, cmap='RdBu_r',clim=[-.6,.6])
pr=axs[1,0].pcolormesh(x,y,zp, cmap='RdBu_r',clim=[-.6,.6])
pr=axs[2,0].pcolormesh(x,y,zs, cmap='RdBu_r',clim=[-.6,.6])
fig.suptitle('LCL )                                                   LCH',fontsize=18,y=0.96)
axs[0,0].spines[['right', 'top']].set_visible(True)
axs[1,0].spines[['right', 'top']].set_visible(True)
axs[2,0].spines[['right', 'top']].set_visible(True)

plt.xlabel('Month')
axs[0,0].scatter(x[sig_temp], y[sig_temp], marker = '.',color='k')
axs[1,0].scatter(x[sig_precip], y[sig_precip], marker = '.',color='k')
axs[2,0].scatter(x[sig_spei], y[sig_spei], marker = '.',color='k')
axs[0,0].set_title('Maximum Temperature')
axs[1,0].set_title('Precipitation')
axs[2,0].set_title('SPEI')

fig.tight_layout(h_pad=1,w_pad=5.5)
axs[0,0].xaxis.set_major_locator(MultipleLocator(1))
axs[1,0].xaxis.set_major_locator(MultipleLocator(1))
axs[2,0].xaxis.set_major_locator(MultipleLocator(1))
axs[0,0].yaxis.set_major_locator(MultipleLocator(1))
axs[1,0].yaxis.set_major_locator(MultipleLocator(1))
axs[2,0].yaxis.set_major_locator(MultipleLocator(1))
axs[0,1].set_yticks(range(11), labels=SeriesLabel)
axs[1,1].set_yticks(range(11), labels=SeriesLabel)
axs[2,1].set_yticks(range(11), labels=SeriesLabel)
axs[0,0].set_xticks(range(1,13), labels=months)
axs[1,0].set_xticks(range(1,13), labels=months)
axs[2,0].set_xticks(range(1,13), labels=months)
cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
plt.subplots_adjust(bottom=0.1, right=.85, top=0.9,left=0.15)
fig.colorbar(r,cax=cbar_ax,label='Correlation (R)',ticks=np.linspace(-.6,.6,num=13))
axs[0,0].text(0, 11, 'a)')
axs[0,1].text(0, 11, 'b)')
axs[1,0].text(0, 11, 'c)')
axs[1,1].text(0, 11, 'd)')
axs[2,0].text(0, 11, 'e)')
axs[2,1].text(0, 11, 'f)')

axs[0,0].add_patch(Rectangle((0.5,-0.5),12,5,fill=False,edgecolor='black'))
axs[0,1].add_patch(Rectangle((0.5,-0.5),12,5,fill=False,edgecolor='black'))
axs[1,0].add_patch(Rectangle((0.5,-0.5),12,5,fill=False,edgecolor='black'))
axs[1,1].add_patch(Rectangle((0.5,-0.5),12,5,fill=False,edgecolor='black'))
axs[2,0].add_patch(Rectangle((0.5,-0.5),12,5,fill=False,edgecolor='black'))
axs[2,1].add_patch(Rectangle((0.5,-0.5),12,5,fill=False,edgecolor='black'))
plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/climatecorr.eps', format='eps')
#plt.close()

#
################################
################### CLIMATE RECONS

JJSpei=pd.Series(np.mean([Spei[Spei['Month']==6][0],Spei[Spei['Month']==7][0]],axis=0))
JJSpei.index=Spei[Spei['Month']==7]['Year']
ASTmax=pd.Series(np.mean([Tmax[Tmax['Month']==8][0],Tmax[Tmax['Month']==9][0]],axis=0))
ASTmax.index=Tmax[Tmax['Month']==9]['Year']
SpringSpei=pd.Series(np.mean([Spei[Spei['Month']==3][0],Spei[Spei['Month']==4][0],Spei[Spei['Month']==5][0],Spei[Spei['Month']==6][0]],axis=0))
SpringSpei.index=Spei[Spei['Month']==3]['Year']

StartYear=1917

climate=ASTmax
cyr=ASTmax.index
proxy=HiSpline20[HiSpline20.index>=StartYear][['Spline30yr_tucson_MXDCWA_q75_20mu_gap.txt','Spline30yr_tucson_MXCWTRAD_q75_20mu_gap.txt']]
#proxy=Hrpcs['RPC1']
pyr=HiSpline20[HiSpline20.index>=StartYear]['Spline30yr_tucson_MXCWTRAD_q75_20mu_gap.txt'].index
calibrate=np.arange(1967,2017,1)
calibrate2=np.arange(StartYear,1967,1)
validate=np.arange(1917,1967,1)
validate2=np.arange(1967,2017,1)

[yhat,s3R2v]=u.simple_cps(climate,cyr,proxy,pyr,calibrate,validate)
[yhat2,s3R2v2]=u.simple_cps(climate,cyr,proxy,pyr,calibrate2,validate2)
c=np.arange(1917,2017,1)
[yhat3,s3R2v3]=u.simple_cps(climate,cyr,proxy,pyr,c,c)


print(s3R2v3)
fig, ax = plt.subplots(3,1,figsize=(5, 7))
og=ax[0].plot(climate,color='black')
#ax[0].set_xlabel('Year')
ax[0].set_ylabel('Maximum Temperature ($^\circ$C)', color='k')
re=ax[0].plot(yhat3,color=[0.8392,0.3765,0.3020])
ax[0].set_ylim(21,29)
ax[0].set_xlim(1917,2017)
ax[0].grid(color=[.5,.5,.5], linestyle='--', linewidth=.2, alpha=0.3)
ax[0].set_xticks(range(1920,2020,10))
ax[0].set_xticklabels(range(1920,2020,10), fontsize=8)
ax[0].spines[['right', 'top']].set_visible(True)
legend=ax[0].legend(['Original','Reconstructed'],frameon = 1,fontsize=8,loc=3)
frame = legend.get_frame()
frame.set_color('white')
ax[0].axvline(x=1967,color=[.2,.2,.2],dashes=(2,1))
ax[0].text(1970, 28.2, 'cal: 1917-1967,  val: 1967:2017',fontsize=6)
ax[0].text(1980, 27.3, 'R$^2$={:.2f}'.format(s3R2v2))
ax[0].text(1930, 28.2, 'cal: 1967-2017,  val: 1917:1967',fontsize=6)
ax[0].text(1940, 27.3, 'R$^2$={:.2f}'.format(s3R2v))
ax[0].text(1918, 29.25, 'a)')
ax[0].set_title('Aug/Sep Max. Temperature - LCH aMXD/MxRCWT',fontsize=10)
#fig.tight_layout(pad=1.5)
#plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/Tmaxrecon_splitvalidate.eps', format='eps')


climate=SpringSpei  
cyr=SpringSpei.index
proxy=LoSpline20[LoSpline20.index>=StartYear]['Spline30yr_tucson_MLA.EW_q75_20mu_gap.txt']
pyr=LoSpline20[LoSpline20.index>=StartYear]['Spline30yr_tucson_MLA.EW_q75_20mu_gap.txt'].index
calibrate=np.arange(1967,2017,1)
calibrate2=np.arange(StartYear,1967,1)
validate=np.arange(1917,1967,1)
validate2=np.arange(1967,2017,1)


[yhat,s3R2v]=u.simple_cps(climate,cyr,proxy,pyr,calibrate,validate)
[yhat2,s3R2v2]=u.simple_cps(climate,cyr,proxy,pyr,calibrate2,validate2)
c=np.arange(1917,2017,1)
[yhat3,s3R2v3]=u.simple_cps(climate,cyr,proxy,pyr,c,c)
print(s3R2v3)

og=ax[1].plot(climate,color='black')
#ax[1].set_xlabel('Year')
ax[1].set_ylabel('SPEI', color='k')
re=ax[1].plot(yhat3,color=[0.8392,0.3765,0.3020])
ax[1].set_ylim(-3,3)
ax[1].set_xlim(1917,2017)
ax[1].grid(color=[.5,.5,.5], linestyle='--', linewidth=.2, alpha=0.3)
ax[1].set_xticks(range(1920,2020,10))
ax[1].set_xticklabels(range(1920,2020,10), fontsize=8)
ax[1].spines[['right', 'top']].set_visible(True)
legend=ax[1].legend(['Original','Reconstructed'],frameon = 1,fontsize=8,loc=3)
frame = legend.get_frame()
frame.set_color('white')
ax[1].axvline(x=1967,color=[.2,.2,.2],dashes=(2,1))
ax[1].text(1970, 2.4, 'cal: 1917-1967,  val: 1967:2017',fontsize=6)
ax[1].text(1980, 1.7, 'R$^2$={:.2f}'.format(s3R2v2))
ax[1].text(1930, 2.4, 'cal: 1967-2017,  val: 1917:1967',fontsize=6)
ax[1].text(1940, 1.7, 'R$^2$={:.2f}'.format(s3R2v))
ax[1].set_title('March-June SPEI - LCL EWLA',fontsize=10)
ax[1].text(1918,3.2, 'b)')
#plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/LoSPEIrecon_fullvalidate.eps', format='eps')
#plt.close()


climate=JJSpei 
cyr=JJSpei.index
proxy=HiSpline20[HiSpline20.index>=StartYear]['Spline30yr_tucson_MLA.EW_q75_20mu_gap.txt']
pyr=HiSpline20[HiSpline20.index>=StartYear]['Spline30yr_tucson_MLA.EW_q75_20mu_gap.txt'].index
#proxy=Hrpcs['RPC2']
calibrate=np.arange(1967,2017,1)
calibrate2=np.arange(StartYear,1967,1)
validate=np.arange(1917,1967,1)
validate2=np.arange(1967,2017,1)


[yhat,s3R2v]=u.simple_cps(climate,cyr,proxy,pyr,calibrate,validate)
[yhat2,s3R2v2]=u.simple_cps(climate,cyr,proxy,pyr,calibrate2,validate2)
c=np.arange(1917,2017,1)
[yhat3,s3R2v3]=u.simple_cps(climate,cyr,proxy,pyr,c,c)
print(s3R2v3)

og=ax[2].plot(climate,color='black')
ax[2].set_xlabel('Year')
ax[2].set_ylabel('SPEI', color='k')
re=ax[2].plot(yhat3,color=[0.8392,0.3765,0.3020])
ax[2].set_ylim(-3,3)
ax[2].set_xlim(1917,2017)
ax[2].grid(color=[.5,.5,.5], linestyle='--', linewidth=.2, alpha=0.3)
ax[2].set_xticks(range(1920,2020,10))
ax[2].set_xticklabels(range(1920,2020,10), fontsize=8)
ax[2].spines[['right', 'top']].set_visible(True)
legend=ax[2].legend(['Original','Reconstructed'],frameon = 1,fontsize=8,loc=3)
frame = legend.get_frame()
frame.set_color('white')
ax[2].axvline(x=1967,color=[.2,.2,.2],dashes=(2,1))
ax[2].text(1970, 2.4, 'cal: 1917-1967,  val: 1967:2017',fontsize=6)
ax[2].text(1980, 1.7, 'R$^2$={:.2f}'.format(s3R2v2))
ax[2].text(1930, 2.4, 'cal: 1967-2017,  val: 1917:1967',fontsize=6)
ax[2].text(1940, 1.7, 'R$^2$={:.2f}'.format(s3R2v))
ax[2].set_title('June/July SPEI - LCH EWLA',fontsize=10)
ax[2].text(1917,3.2, 'c)')
fig.tight_layout(pad=1.2)
plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/AllRecons.eps', format='eps')
#plt.close()

###########################################
#### SNOW CORRELATIONS ################

Workdir='/Users/julieedwards/Documents/Projects/LCP/climate_data/snow/'
os.chdir(Workdir)

snowf='wateryearSNOTEL.csv'
snotel=pd.read_csv(snowf,index_col=None,header=None)
snotelyears=range(1981,2018)


snotel.index=['o','n','d','J','F','Mr','A','M','Jn','Jl','Au','s']
snotel.columns=list(snotelyears)
StartYear=1981
EndYear=2017
snow_p=[]
snow_r=[]
rut_p=[]
rut_r=[]
period=30#Spline 50% freq cutoff
############## LCL ###################
for index in snotel.index:
    for j in range(len(Series)+4):
        Snowdetr = snotel.loc[index]
        r_snow,p_snow=pearsonr(Snowdetr,LoSpline20[LoSpline20.index>=StartYear][LoSpline20.columns[j]], alternative='two-sided')
        snow_r.append(r_snow)
        snow_p.append(p_snow)
        r_rutsnow,p_rutsnow=pearsonr(Snowdetr,HiSpline20[HiSpline20.index>=StartYear][HiSpline20.columns[j]], alternative='two-sided')
        rut_r.append(r_rutsnow)
        rut_p.append(p_rutsnow)
   

SnoR = np.array(snow_r).reshape(12,len(Series)+4).transpose()
SnoP = np.array(snow_p).reshape(12,len(Series)+4).transpose()
RutR = np.array(rut_r).reshape(12,len(Series)+4).transpose()
RutP = np.array(rut_p).reshape(12,len(Series)+4).transpose()

SnoR=SnoR.T[1:8].T
SnoP=SnoP.T[1:8].T
RutR=RutR.T[1:8].T
RutP=RutP.T[1:8].T
############### PLOTTING
SeriesLabel=["RPC2","RPC1","PC2","PC1","aMXD","MxRCWT","MxTCWT","LWW","EWLA","EWW","TRW"]
sig_snow=np.where(SnoP < 0.01)

months=['n','d','J','F','M','A','M']

dx, dy = 1, 1
y, x = np.mgrid[0:10+dy:dy, 1:7+dx:dx]
z=SnoR
fig, axs = plt.subplots(1,2,figsize=(7.5, 2.5),constrained_layout = True)
r=axs[0].pcolormesh(x,y,z, cmap='RdBu_r',clim=[-.6,.6])
axs[0].text(0,11, 'a)')
axs[0].spines[['right', 'top']].set_visible(True)
axs[0].set_xlabel('Month')
axs[0].scatter(x[sig_snow], y[sig_snow], marker = '.',color='k')
axs[0].set_title('LCL')
axs[0].axhline(y=4.5,color='k')
fig.tight_layout(pad=0)
axs[0].xaxis.set_major_locator(MultipleLocator(1))
axs[0].set_yticks(range(11), labels=SeriesLabel)
axs[0].set_xticks(range(1,8), labels=months)
#plt.subplots_adjust(bottom=0.1, right=.6, top=0.9,left=0.3)
z=RutR
sig_snow=np.where(RutP < 0.01)
r=axs[1].pcolormesh(x,y,z, cmap='RdBu_r',clim=[-.6,.6])
axs[1].text(0,11, 'b)')
axs[1].axhline(y=4.5,color='k')
axs[1].spines[['right', 'top']].set_visible(True)
plt.xlabel('Month')
axs[1].scatter(x[sig_snow], y[sig_snow], marker = '.',color='k')
axs[1].set_title('LCH')
axs[1].xaxis.set_major_locator(MultipleLocator(1))
axs[1].set_yticks(range(11), labels=[])
axs[1].set_xticks(range(1,8), labels=months)
cbar_ax = fig.add_axes([1, .175, 0.01, 0.745])
fig.colorbar(r,cax=cbar_ax,label='Correlation (R)',ticks=np.linspace(-.6,.6,num=7))
plt.subplots_adjust(left=0.5)
fig.suptitle('SNOTEL (1981-2017)',y=1.1,x=.75)
plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/snotelcorr.eps', format='eps',bbox_inches = 'tight')
plt.close()








################################