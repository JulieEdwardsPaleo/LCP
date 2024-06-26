

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
import math
from sklearn.preprocessing import StandardScaler




###########################################
#############   DATA READ IN 
###########################################

def read_climate_data(workdir, filename, decode_times=False):
    filepath = os.path.join(workdir, filename)
    with xr.open_dataarray(filepath, decode_times=decode_times).load() as data_array:
        units, reference_date = data_array.time.attrs['units'].split('since')
        data_array['time'] = pd.date_range(start=reference_date, periods=data_array.sizes['time'], freq='M')
    return data_array

def extract_time_components(data_array):
    year = data_array['time'].dt.strftime('%Y').values
    month = data_array['time'].dt.strftime('%m').values
    return year, month

climate_data_dir = '/Users/julieedwards/Documents/Projects/LCP/climate_data/cru/'
climate_variables = {
    'Tmean_CRU': 'icru4_tmp_-105.22E_36.83N_n.nc',
    'Tmax_CRU': 'icru4_tmx_-105.22E_36.83N_n.nc',
    'Precip_CRU': 'icru4_pre_-105.22E_36.83N_n.nc',
    'Spei_CRU': 'ispei_01_-105.22E_36.83N_n.nc'
}

# Read and process climate data
climate_data = {}
for var_name, filename in climate_variables.items():
    data_array = read_climate_data(climate_data_dir, filename)
    climate_data[var_name] = data_array
    
    # Extract year and month for Tmean_CRU and Spei_CRU
    if var_name in ['Tmean_CRU', 'Spei_CRU']:
        year, month = extract_time_components(data_array)
        climate_data[f'{var_name}_Year'] = year
        climate_data[f'{var_name}_Month'] = month

def read_csv_files(workdir, filenames):
    os.chdir(workdir)
    dataframes = {}
    for filename in filenames:
        df = pd.read_csv(filename, index_col=None, header=None)
        dataframes[filename] = df
    return dataframes


def process_qwa_data(workdir, series, reso, method):
    os.chdir(workdir)
    files = os.listdir(workdir)
    relevant_files=[]
    for filename in files:
        for i in range(len(series)):
            if fnmatch.fnmatch(filename,f"Spline30*{series[i]}*{method}*{reso}*"):
                relevant_files.append(filename)
    relevant_files=set(relevant_files)
    li=[]
    df=[]
    for filename in relevant_files:
        df =pd.read_csv(filename,index_col=None, header=0)
        li.append(df["std"])
    frame=pd.concat(li,axis=1,ignore_index=True)
    frame.columns = relevant_files
    frame=frame.set_index(df["Unnamed: 0"])
    return frame

# TopoWx Temperature data
climate_data_dir = '/Users/julieedwards/Documents/Projects/LCP/climate_data/'
topo_files = ['LCH_topo_tmean.csv', 'LCHtopoMonthly.csv', 'LCLtopoMonthly.csv', 'TotalWaterYearPrecip.csv']
topo_data = read_csv_files(climate_data_dir, topo_files)

# QWA Data - 30 year spline and linear detrend
series = ["MXCWTRAD", "MXDCWA", "EWW", "MRW", "LWW", "MXCWTTAN", "MLA.EW"]
reso = "20mu"
method = "q75"
qwa_high_dir= '/Users/julieedwards/Documents/Projects/LCP/QWA_High/concat/Summary/'
uHiSpline20 = process_qwa_data(qwa_high_dir, series, reso, method)

qwa_low_dir = '/Users/julieedwards/Documents/Projects/LCP/QWA_Low/concat/Summary/'
uLoSpline20 = process_qwa_data(qwa_low_dir, series, reso, method)



##########################################
################## CLIMATOLOGY
################################################

def prepare_climatology_data(data_array):
    climatology_mean = data_array.groupby("time.month").mean("time")
    climatology_std = data_array.groupby("time.month").std("time")
    return climatology_mean, climatology_std

def plot_climatology(ax, x, y_mean, y_std, color, label, ylabel, ylim, linestyle='-', fill=True):
    ax.plot(x, y_mean, color=color, linestyle=linestyle, label=label)
    if fill:
        ax.fill_between(x, y_mean-y_std, y_mean+y_std, alpha=0.2, color=color)
    ax.set_ylabel(ylabel, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim(ylim)
    ax.set_xticks(x)  


def generate_climatology_plot(Tmean_CRU, Precip_CRU, topoMH, topoML):
    fig, ax1 = plt.subplots(figsize=(4, 3))
    plt.title('Little Costilla Peak climate')
    ax1.set_xlabel('Month')

    T_climatology, T_std = prepare_climatology_data(Tmean_CRU)
    P_climatology, P_std = prepare_climatology_data(Precip_CRU)

    plot_climatology(ax1, T_climatology['month'], T_climatology, T_std, 
                     color=[0.8392, 0.3765, 0.3020], label=None, 
                     ylabel='Temperature ($^\circ$C)', ylim=(-15, 50))
    ax1.plot(T_climatology["month"], topoMH, color='black', ls='--', label='LCH TopoWx')
    ax1.plot(T_climatology["month"], topoML, color='black', ls=':', label='LCL TopoWx')
    ax1.legend(loc='upper left', fontsize=8).get_frame().set_edgecolor('w')
    ax1.grid(color='gray', linestyle='--', linewidth=0.5)
    ax1.set_xlim(1, 12)

    ax2 = ax1.twinx()
    plot_climatology(ax2, P_climatology['month'], P_climatology, P_std,
                     color=[0.1294, 0.4000, 0.6745], label=None,
                     ylabel='Precipitation (mm)', ylim=(-30, 100), fill=True)
    fig.tight_layout()  
    plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/climatology.eps', format="eps", transparent=True)



Tmean_CRU = climate_data['Tmean_CRU']
Precip_CRU = climate_data['Precip_CRU']

topoMH = topo_data['LCHtopoMonthly.csv'].iloc[:, 0]  
topoML = topo_data['LCLtopoMonthly.csv'].iloc[:, 0]
months = range(1, 13)  
topoMH = pd.Series(data=topoMH.values, index=months)
topoML = pd.Series(data=topoML.values, index=months)
months = range(1, 13)  
generate_climatology_plot(Tmean_CRU, Precip_CRU, topoMH, topoML)


##########################################
################### PCA  
###########################################
SeriesOrder=["Spline30yr_gap_MRW_q75_20mu.txt","Spline30yr_gap_EWW_q75_20mu.txt","Spline30yr_gap_MXCWTRAD_q75_20mu.txt","Spline30yr_gap_MXDCWA_q75_20mu.txt","Spline30yr_gap_MXCWTTAN_q75_20mu.txt","Spline30yr_gap_LWW_q75_20mu.txt","Spline30yr_gap_MLA.EW_q75_20mu.txt"]
HiSpline20=uHiSpline20[SeriesOrder]
LoSpline20=uLoSpline20[SeriesOrder]

def PCAbi(df):
    df=df[df.index>1916]
    scaling=StandardScaler()
    scaling.fit(df)
    Scaled_data=scaling.transform(df)
    C = np.corrcoef(Scaled_data,rowvar=False,ddof=1)
    U,S,V = np.linalg.svd(C)
    A = U[:,0:2] 
    model = pca(normalize=True, n_components=2)
    results=model.fit_transform(df)
    exvar=results['explained_var']
    PCs = df @ A
    PCs.columns=['PC1','PC2']
    Reofs, rotation_matrix = u.varimax_rotation(A,normalize=True)
    RPCs = pd.DataFrame(Scaled_data @ Reofs)
    RPCs.index=df.index
    RPCs.columns=['RPC1','RPC2']

    return U, exvar, PCs, RPCs, Reofs

def biplot(ax, U, exvar, colors, labels, title, panel_label,rotated):
    ax.hlines(0,-1,1,'k')
    ax.vlines(0,-1,1,'k')
    for i in range(len(labels)):
        ax.plot([0,U[i,0]*-1],[0,U[i,1]*-1],markersize=10,color=colors[i])
        ax.plot(U[i,0]*-1,U[i,1]*-1,'o',markersize=8,color=colors[i], markeredgecolor='k',
         markeredgewidth=0.3) 
        ax.text(U[i,0]*-1,U[i,1]*-1,labels[i],fontsize=7)

    ax.spines[['right', 'top']].set_visible(True)    

    if rotated==False:
        ax.set_xlabel(f'PC1: {exvar[0] * 100:.1f}%')
        ax.set_ylabel(f'PC2: {(exvar[1]-exvar[0]) * 100:.1f}%')
    else:
        ax.set_xlabel('RPC1')
        ax.set_ylabel('RPC2')       
    ax.set_title(title, fontsize=10,y=1)
    ax.text(-0.8, 0.95, panel_label, fontsize=10)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim(-.9, .9)
    ax.set_ylim(-.9, .9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticks(np.linspace(-.8,.8,5))
    ax.set_yticks(np.linspace(-.8,.8,5))
    latewood=ax.plot([], [], 'o',c=(0.2627,0.5765,0.7647), markersize=8,label='LW')
    earlywood=ax.plot([], [], 'o',c=(0.8392,0.3765,0.3020), markersize=8,label='EW')
    full=ax.plot([], [], 'o',c='k', markersize=8,label='TRW')
    leg=ax.legend(loc='lower left')
    leg.get_frame().set_edgecolor('w')

U_hi,exvar_hi,PCs_hi,RPCs_hi, Reofs_hi=PCAbi(HiSpline20)
U_lo,exvar_lo,PCs_lo,RPCs_lo, Reofs_lo=PCAbi(LoSpline20)

colors=[(0,0,0),(0.8392,0.3765,0.3020),(0.2627,0.5765,0.7647),(0.2627,0.5765,0.7647),(0.2627,0.5765,0.7647),(0.2627,0.5765,0.7647),(0.8392,0.3765,0.3020)]
SeriesLabel=["TRW","EWW","MxRCWT","aMXD","MxTCWT","LWW","EWLA"]

fig, axs = plt.subplots(2, 2, figsize=(6, 5))

biplot(axs[0, 0], U_lo, exvar_lo, colors, SeriesLabel, 'LCL QWA PCA', 'a)',rotated=False)
biplot(axs[0, 1], U_hi, exvar_hi, colors, SeriesLabel, 'LCH QWA PCA', 'b)',rotated=False)
biplot(axs[1, 0], Reofs_lo, ['',''], colors, SeriesLabel, 'LCL QWA Rotated PCA', 'c)',rotated=True)
biplot(axs[1, 1], Reofs_hi, ['',''], colors, SeriesLabel, 'LCH QWA Rotated PCA', 'd)',rotated=True)

fig.tight_layout(w_pad=1,h_pad=0.8)
plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/PCA_ALL.eps', format='eps')
 

##################################
####### Climate correlations
###################################

def prep_climate(var,startyear,endyear):
    output_df=pd.DataFrame(climate_data[var])
    if var=='Spei_CRU':
        output_df['Month']=climate_data['Spei_CRU_Month'].astype(np.int64)
        output_df['Year']=climate_data['Spei_CRU_Year'].astype(np.int64)  
    else:
        output_df['Month']=climate_data['Tmean_CRU_Month'].astype(np.int64)
        output_df['Year']=climate_data['Tmean_CRU_Year'].astype(np.int64)
    
    output_df=output_df[(output_df['Year']>=startyear)&(output_df['Year']<=endyear)]
    return output_df

Tmax=prep_climate('Tmax_CRU',1917,2017)
Precip=prep_climate('Precip_CRU',1917,2017)
Spei=prep_climate('Spei_CRU',1917,2017)

def prep_QWA(QWAchronos,PCs,RPCs,seriesorder):
    output=pd.concat([QWAchronos,PCs*-1,RPCs*-1],axis=1)
    output=output[seriesorder]
    return output

SeriesOrder=['RPC2', 'PC2','Spline30yr_gap_MLA.EW_q75_20mu.txt','Spline30yr_gap_EWW_q75_20mu.txt','Spline30yr_gap_MRW_q75_20mu.txt', 'RPC1','PC1','Spline30yr_gap_MXDCWA_q75_20mu.txt','Spline30yr_gap_MXCWTRAD_q75_20mu.txt','Spline30yr_gap_MXCWTTAN_q75_20mu.txt','Spline30yr_gap_LWW_q75_20mu.txt']

HiSpline20=prep_QWA(uHiSpline20,PCs_hi,RPCs_hi,SeriesOrder)
LoSpline20=prep_QWA(uLoSpline20,PCs_lo,RPCs_lo,SeriesOrder)



def Climate_corr(climate_input,QWA_input,period,startyear):
    r=[]
    p=[]
    for i in range(0,12):
        for j in range(len(QWA_input.columns)):
            detrended=u.spline(climate_input[climate_input['Month']==i+1]['Year'],climate_input[climate_input['Month']==i+1][0],period)
            r_temp,p_temp=pearsonr(detrended,QWA_input[QWA_input.index>=startyear][QWA_input.columns[j]], alternative='two-sided')
            r.append(r_temp)
            p.append(p_temp)
    output_r=np.array(r).reshape(12,len(QWA_input.columns)).transpose()
    output_p=np.array(p).reshape(12,len(QWA_input.columns)).transpose()
    return output_r,output_p

Tmax_r_hi,Tmax_p_hi=Climate_corr(Tmax,HiSpline20,30,1917)
Tmax_r_lo,Tmax_p_lo=Climate_corr(Tmax,LoSpline20,30,1917)
Precip_r_hi,Precip_p_hi=Climate_corr(Precip,HiSpline20,30,1917)
Precip_r_lo,Precip_p_lo=Climate_corr(Precip,LoSpline20,30,1917)
Spei_r_hi,Spei_p_hi=Climate_corr(Spei,HiSpline20,30,1917)
Spei_r_lo,Spei_p_lo=Climate_corr(Spei,LoSpline20,30,1917)


def CC_plot(ax,r,p, alpha,climate_label,serieslabel,panel):
    sig=np.where(p<alpha)
    months=["J","F","M","A","M","J","J","A","S","O","N","D"]
    dx,dy=1,1
    y,x=np.mgrid[0:len(serieslabel)-1+dy:dy, 1:12+dx:dx]
    r=ax.pcolormesh(x,y,r,cmap='RdBu_r',clim=[-.6,.6])
    ax.spines[['right','top']].set_visible(True)

    ax.scatter(x[sig],y[sig],marker='.',color='k')
    ax.axhline(y = 4.5, color = 'k', linestyle = '-') 
    ax.set_title(climate_label)
    ax.text(0, 11, panel)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_yticks(range(11), labels=SeriesLabel)
    ax.set_xticks(range(1,13), labels=months)
    return r


############### PLOTTING
SeriesLabel=['RPC2', 'PC2','EWLA','EWW','TRW', 'RPC1','PC1','aMXD','MxRCWT','MxTCWT','LWW']

fig, axs = plt.subplots(3, 2,figsize=(7.5, 7.5))
CC_plot(axs[0,0],Tmax_r_lo,Tmax_p_lo,0.01,'Maximum Temperature',SeriesLabel,'a)')
CC_plot(axs[1,0],Precip_r_lo,Precip_p_lo,0.01,'Precipitation',SeriesLabel,'b)')
CC_plot(axs[2,0],Spei_r_lo,Spei_p_lo,0.01,'SPEI',SeriesLabel,'c)')
CC_plot(axs[0,1],Tmax_r_hi,Tmax_p_hi,0.01,'Maximum Temperature',SeriesLabel,'d)')
CC_plot(axs[1,1],Precip_r_hi,Precip_p_hi,0.01,'Precipitation',SeriesLabel,'e)')
r=CC_plot(axs[2,1],Spei_r_hi,Spei_p_hi,0.01,'SPEI',SeriesLabel,'f)')
axs[2,1].set_xlabel('Month')
axs[2,0].set_xlabel('Month')

fig.tight_layout(h_pad=1,w_pad=2)
cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
fig.colorbar(r,cax=cbar_ax,label='Correlation (R)',ticks=np.linspace(-.6,.6,num=13))
plt.subplots_adjust(bottom=0.1, right=.85, top=0.9,left=0.15)
fig.suptitle('LCL                                 LCH',fontsize=18,y=0.96)
plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/climatecorr.eps', format='eps')



######################################
################ CLIMATE RECONS
############################################

JJSpei=pd.Series(np.mean([Spei[Spei['Month']==6][0],Spei[Spei['Month']==7][0]],axis=0))
JJSpei.index=Spei[Spei['Month']==7]['Year']
JSpei=pd.Series(Spei[Spei['Month']==6][0])
JSpei.index=Spei[Spei['Month']==6]['Year']
ASTmax=pd.Series(np.mean([Tmax[Tmax['Month']==8][0],Tmax[Tmax['Month']==9][0]],axis=0))
ASTmax.index=Tmax[Tmax['Month']==9]['Year']
SpringSpei=pd.Series(np.mean([Spei[Spei['Month']==3][0],Spei[Spei['Month']==4][0],Spei[Spei['Month']==5][0],Spei[Spei['Month']==6][0]],axis=0))
SpringSpei.index=Spei[Spei['Month']==3]['Year']

def recon(climate, cyr,proxy,pyr,fullperiod):
    #climate=u.spline(cyr,climate,30)
    fullyhat,fullR2v,fullR2c,fullre,fullce=u.simple_cps(climate,cyr,proxy,pyr,fullperiod,fullperiod)
    center = np.take(fullperiod, fullperiod.size // 2)
    period1=np.arange(fullperiod[0],center+1,1)
    period2=np.arange(center,fullperiod[-1]+1,1)
    eyhat,earlyvalR2,eR2c,eRE,eCE=u.simple_cps(climate,cyr,proxy,pyr,period2,period1)
    lyhat,latevalR2,lR2c,lRE,lCE=u.simple_cps(climate,cyr,proxy,pyr,period1,period2)
    err = math.sqrt(np.mean((climate-fullyhat)**2))
    output = [err,fullR2v,fullR2c,fullre,fullce, earlyvalR2,eR2c,eRE,eCE,latevalR2,lR2c,lRE,lCE]
    columns = ['RMSE','fullR2v', 'fullR2c', 'fullre', 'fullce', 'earlyvalR2', 'eR2c', 'eRE', 'eCE', 'latevalR2', 'lR2c', 'lRE', 'lCE']
    output_df = pd.DataFrame([output],columns=columns)
    return fullyhat,output_df


StartYear=1917
fullperiod=np.arange(1917,2018,1)

ASTyhat,ASToutput=recon(ASTmax, ASTmax.index,HiSpline20[HiSpline20.index>=StartYear][['Spline30yr_gap_MXDCWA_q75_20mu.txt','Spline30yr_gap_MXCWTRAD_q75_20mu.txt']],HiSpline20[HiSpline20.index>=StartYear]['Spline30yr_gap_MXCWTRAD_q75_20mu.txt'].index,fullperiod)
Springyhat,SPRINGoutput=recon(SpringSpei, SpringSpei.index,LoSpline20[LoSpline20.index>=StartYear]['Spline30yr_gap_MLA.EW_q75_20mu.txt'],LoSpline20[LoSpline20.index>=StartYear]['Spline30yr_gap_MLA.EW_q75_20mu.txt'].index,fullperiod)
JJyhat,JJoutput= recon(JJSpei,JJSpei.index,HiSpline20[HiSpline20.index>=StartYear]['Spline30yr_gap_MLA.EW_q75_20mu.txt'],HiSpline20[HiSpline20.index>=StartYear]['Spline30yr_gap_MLA.EW_q75_20mu.txt'].index,fullperiod)

Jyhat,Joutput= recon(JSpei,JSpei.index,LoSpline20[LoSpline20.index>=StartYear]['Spline30yr_gap_MRW_q75_20mu.txt'],LoSpline20[LoSpline20.index>=StartYear]['Spline30yr_gap_MRW_q75_20mu.txt'].index,fullperiod)




#######################################################
###### DROUGHT Zoom-in Timeseries


def panels(axes,xlim,xticks):
    ax0=axes.inset_axes([0, 0.9, 1, 0.23])
    ax0.spines['bottom'].set_visible(False)
    ax0.tick_params(
        axis='both',          
        which='both',      
        bottom=False,      
        top=False, left=False,
        labelbottom=False)
    ax0.set_xlim(xlim)
    ax0.set_yticks([])
    ax0.set_xticks(xticks)
    ax0.set_xticklabels([]) 
    ax0.set_yticklabels([]) 
    ax0.grid(True,ls=':',color=[.8,.8,.8])
    ax1 = axes.inset_axes([0, 0.75, 1, 0.35])
    ax1.plot(ASTmax,color='k',linewidth=2)
    ax1.plot(ASTyhat,color=[0.2627,0.5765,0.7647],linewidth=2)
    ax1.fill_between(ASTyhat.index, ASTyhat - ASToutput['RMSE'].iloc[0], ASTyhat + ASToutput['RMSE'].iloc[0], color=[0.2627,0.5765,0.7647],alpha=0.2)
    ax1.set_xlim(xlim)
    ax1.set_ylim([22, 28])
    ax1.set_facecolor('none')
    ax1.set_xticklabels([]) 
    ax1.grid(True,ls=':',color=[.8,.8,.8])
    ax1.set_xticks(xticks)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2 = axes.inset_axes([0, 0.4, 1, 0.35])
    ax2.axhline(y=0,color=[0.8,0.8,0.8],ls='--')
    ax2.plot(SpringSpei, 'k', linewidth=2)
    ax2.plot(Springyhat, color=[0.8392,0.3765,0.3020], linewidth=2)
    ax2.fill_between(Springyhat.index, Springyhat - SPRINGoutput['RMSE'].iloc[0], Springyhat + SPRINGoutput['RMSE'].iloc[0], color=[0.8392,0.3765,0.3020],alpha=0.2)
    ax2.set_xlim(xlim)
    ax2.set_ylim([-3, 2])
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('right')
    ax2.set_facecolor('none')
    ax2.spines[['top','bottom']].set_visible(False)
    ax2.grid(True,ls=':',color=[.8,.8,.8])
    ax2.set_xticks(xticks)
    ax3 = axes.inset_axes([0.0, 0.12, 1, 0.35])
    ax3.axhline(y=0,color=[0.8,0.8,0.8],ls='--')
    ax3.plot(JJSpei, 'k', linewidth=2)
    ax3.plot(JJyhat, color=[0.698,0.1215,0.1725], linewidth=2)
    ax3.fill_between(JJyhat.index, JJyhat - JJoutput['RMSE'].iloc[0], JJyhat + JJoutput['RMSE'].iloc[0], color=[0.698,0.1215,0.1725],alpha=0.2)
    ax3.set_xlim(xlim)
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticks,fontsize=7)
    ax3.set_ylim([-3, 2])
    ax3.spines['top'].set_visible(False)
    ax3.set_facecolor('none')
    ax3.grid(True,ls=':',color=[.8,.8,.8])
    if axes==axs[0]:
        ax1.set_ylabel('Aug/Sep T$_\mathrm{max}$ ($^\circ$C)')
        ax1.set_yticklabels([22,24,26,28])
        ax1.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        top=False, 
        labelbottom=False)
        ax3.set_ylabel('Jun/Jul SPEI',x=-1)
    else:
        ax1.set_ylabel(None)
        ax3.set_ylabel(None)
        ax1.set_yticklabels([])
        ax3.set_yticklabels([])
        ax1.tick_params(
        axis='both',          
        which='both',      
        bottom=False,     
        top=False, 
        left=False,                
        labelbottom=False)
        ax3.tick_params(
        axis='y',         
        which='both',     
        bottom=False,   
        top=False, 
        left=False,                
        labelbottom=False)
    if axes==axs[3]:
        ax2.set_ylabel('Spring SPEI') 
        ax2.tick_params(
        axis='x',        
        which='both',   
        bottom=False,    
        top=False,
        right=False, 
        left=False,      
        labelbottom=False)
    else:
        ax2.tick_params(
        axis='both',       
        which='both',     
        bottom=False,  
        top=False,
        right=False, 
        left=False,      
        labelbottom=False)
        ax2.set_xticklabels([]) 
        ax2.set_yticklabels([])
        ax2.set_xticklabels([]) 
fig, axs = plt.subplots(1, 4,figsize=(6, 3),sharex=True,sharey=True)
axs[0].patch.set_facecolor('white')
axs[0].text(-0.055, 0.072, 'a)', fontsize=8)
axs[0].set_yticks([])
axs[0].set_xticks([])
axs[0].spines[['left','right','top','bottom']].set_visible(False)
axs[0].set_title('Dust Bowl',fontsize=9,y=1.125)
panels(axs[0],[1929, 1941],[1930,1935,1940])
axs[1].patch.set_facecolor('white')
axs[1].text(-0.055, 0.072, 'b)', fontsize=8)
axs[1].set_yticks([])
axs[1].set_xticks([])
axs[1].spines[['left','right','top','bottom']].set_visible(False)
axs[1].set_title('Southwest',fontsize=9,y=1.125)
panels(axs[1],[1949, 1961],[1950,1955,1960])
axs[2].text(-0.055, 0.072, 'c)', fontsize=8)
axs[2].patch.set_facecolor('white')
axs[2].set_yticks([])
axs[2].set_xticks([])
axs[2].spines[['left','right','top','bottom']].set_visible(False)
axs[2].set_title("1980's Pluvial",fontsize=9,y=1.125)
panels(axs[2],[1979, 1991],[1980,1985,1990])
axs[3].patch.set_facecolor('white')
axs[3].text(-0.055, 0.072, 'd)', fontsize=8)
axs[3].set_yticks([])
axs[3].set_xticks([])
axs[3].spines[['left','right','top','bottom']].set_visible(False)
axs[3].set_title('Megadrought', fontsize=9,y=1.125)
panels(axs[3],[1999, 2015],[2000,2005,2010,2015])

ob=axs[0].plot([], [], color='k',label='Observation')
t=axs[0].plot([],[],color=[0.2627,0.5765,0.7647],label='Scaled proxy data')
lcl=axs[0].plot([],[],color=[0.8392,0.3765,0.3020])
lch=axs[0].plot([],[],color=[0.698,0.1215,0.1725])

fig.legend(ob,['Observation'],frameon=False,loc='upper left',fontsize=8,bbox_to_anchor=(0.095,.96),handlelength=1.5,handletextpad=0.25)
fig.legend(t,['aMXD/MxRCWT'],frameon=False,loc='upper left',fontsize=8,bbox_to_anchor=(0.095,.67),handlelength=1.5,handletextpad=0.25)
fig.legend(lcl,['LCL EWLA'],frameon=False,loc='upper left',fontsize=8,bbox_to_anchor=(0.095,.45),handlelength=1.5,handletextpad=0.25)
fig.legend(lch,['LCH EWLA'],frameon=False,loc='upper left',fontsize=8,bbox_to_anchor=(0.095,.19),handlelength=1.5,handletextpad=0.25)

plt.tight_layout(pad=0)
fig.supxlabel("Year",y=-0.03,fontsize=10)

plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/DroughtPanels.eps', transparent=False,bbox_inches = 'tight')
plt.show()



###########################################
############# SNOW CORRELATIONS 
####################################


snotel = pd.read_csv('/Users/julieedwards/Documents/Projects/LCP/climate_data/snow/wateryearSNOTEL.csv', index_col=None, header=None)
snotelyears=range(1981,2018)
snotel.index=['o','n','d','J','F','Mr','A','M','Jn','Jl','Au','s']
snotel.columns=list(snotelyears)
StartYear=1981
EndYear=2017
period=30


def snow_corr(climate_input,QWA_input,period,startyear):
    r=[]
    p=[]
    for i in snotel.index:
        for j in range(len(QWA_input.columns)):
            detrended=u.spline(climate_input[climate_input.index==i].columns,climate_input.loc[i],period)
            r_temp,p_temp=pearsonr(detrended,QWA_input[QWA_input.index>=startyear][QWA_input.columns[j]], alternative='two-sided')
            r.append(r_temp)
            p.append(p_temp)
    output_r=np.array(r).reshape(12,len(QWA_input.columns)).transpose()
    output_p=np.array(p).reshape(12,len(QWA_input.columns)).transpose()
    output_r=output_r.T[1:8].T
    output_p=output_p.T[1:8].T
    return output_r,output_p

snow_r_hi,snow_p_hi=snow_corr(snotel,HiSpline20,30,1981)
snow_r_lo,snow_p_lo=snow_corr(snotel,LoSpline20,30,1981)


def snowplot(ax,r,p,site_label,panel):
    sig_snow=np.where(p<0.01)
    dx,dy=1,1
    y,x=np.mgrid[0:10+dy:dy,1:7+dx:dx]
    r=ax.pcolormesh(x,y,r,cmap='RdBu_r',clim=[-.6,.6])
    ax.text(0,11,panel)
    ax.spines[['right', 'top']].set_visible(True)
    ax.set_xlabel('Month')
    ax.scatter(x[sig_snow],y[sig_snow],marker='.',color='k')
    ax.set_title(site_label)
    ax.axhline(y=4.5,color='k')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(range(1,8), labels=months)  
    if panel=='a)':
        ax.set_yticks(range(11), labels=SeriesLabel,fontsize=10)
    else:
        ax.set_yticks(range(11), labels=[])

SeriesLabel=['RPC2', 'PC2','EWLA','EWW','TRW', 'RPC1','PC1','aMXD','MxRCWT','MxTCWT','LWW']
months=['n','d','J','F','M','A','M']
fig, ax = plt.subplots(1,2,figsize=(7.4, 2.5),constrained_layout = True)
snowplot(ax[0],snow_r_lo,snow_p_lo,'LCL','a)')
snowplot(ax[1],snow_r_hi,snow_p_hi,'LCH','b)')

fig.tight_layout(pad=0,w_pad=1)
cbar_ax = fig.add_axes([1.01, .175, .01, .745])
fig.colorbar(r,cax=cbar_ax,label='Correlation (R)',ticks=np.linspace(-.6,.6,num=7))
plt.subplots_adjust(left=0.5)
fig.suptitle('SNOTEL (1981-2017)',y=1.08,x=.75)
plt.savefig('/Users/julieedwards/Documents/Projects/LCP/Figures/snotelcorr.eps', format='eps',bbox_inches = 'tight')


