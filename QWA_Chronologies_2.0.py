######### CODE TO CALL ALL SCRIPTS NEEDED TO TAKE ROXAS OUTPUT TO STANDARD TREE-RING CHRONOLOGY FORMAT ############### Python 3.10.9

import subprocess #For calling other code
import os #to list files in directories
import fnmatch
import pandas as pd


######### TO DO before running #################################################################

#have to make _YearsToExclude.txt in Workdir2

#have to make pith offset file for rcs code in Script 5 if using RCS, match IDs to tucson_ files output from Script 4


#################### CHANGE THESE ###############################################################
#### setting paths and variables for QWA code
Workdir1="/Users/julieedwards/Documents/Projects/LCP/QWA_High/" #individual image output
Workdir2="/Users/julieedwards/Documents/Projects/LCP/QWA_High/concat/"  #concatenated output
Workdir3="/Users/julieedwards/Documents/Projects/LCP/QWA_High/concat/Summary/" # summary output
Workdir4="/Users/julieedwards/Documents/Projects/LCP/QWA_High/concat/Summary/" #final chronologies
yrstart="1000"
yrend="2022"
method="q75" #  # 4 options: "mean","median","q25","q75"   #different methods for aggregation within each intra-ring band  
resolution="20" #spatial resolution of profiles (micrometers), i.e. width of intra-ring bands
Series= "MRW","MXDCWA","MXCWTRAD","MXCWTTAN","EWW","MLA.EW","LWW" #Cell series of interest to use in chronology making

SECTORAGGREGATE="FALSE"   #SLOW, when this option is activated, intra-ring sector profiles are calculated and saved to file
NUMSECTOR="5"   #number of intra-ring sectors (cell assignment based on relative position in the ring)

  

###################### CALL R SCRIPTS ################################ won't have to change anything below this line
# run scripts 1 and 2
res = subprocess.call(["/usr/local/bin/Rscript", "/Users/julieedwards/Documents/Projects/QWAcode/1_SummarizeRadialWedgeData_1.7.R",  Workdir1,Workdir2]) #aggregates data across images to create a single continuous dataset for each section
QCMODE="TRUE" ##when this option is activated, only the intra-annual profile of ring width together with several features useful for quality control are plotted; also the extraction and plotting of annual statistics is skipped in this mode
#See R script 2_CreateIntraanualProfiles for flag meanings
SUPPRESSPLOTTING ="FALSE"  #must be FALSE when QC = TRUE, when this option is activated, no plots are produced (speeds up!)
res = subprocess.call(["/usr/local/bin/Rscript", "/Users/julieedwards/Documents/Projects/QWAcode/2_CreateIntraannualProfiles_1.11.R",  Workdir2,yrstart,yrend,method,resolution,QCMODE,SECTORAGGREGATE,NUMSECTOR,SUPPRESSPLOTTING]) #Bins cell data, QC mode first, does not output usable data, must QC first!
#quality check data after script 2, stop this script, fix any issues and re-run. if good, proceed
QC_check = input('Quality check complete? Ready to proceed? [y/n]')  #go to Workdir2 and look at QC plots,#See R script 2_CreateIntraanualProfiles for flag meanings
if QC_check == 'y':
    QCMODE="FALSE"
    SUPPRESSPLOTTING ="TRUE"
else:
    print('fix errors and rerun this script')
    exit()
# run script 2 for real, QC check off
res = subprocess.call(
 ["/usr/local/bin/Rscript", "/Users/julieedwards/Documents/Projects/QWAcode/2_CreateIntraannualProfiles_1.11.R",  Workdir2,yrstart,yrend,method,resolution,QCMODE,SECTORAGGREGATE,NUMSECTOR,SUPPRESSPLOTTING]) #re-running scripts to output usable data
# run script 3, summarizes different measurements into time series per series, makes it easier to then input into dplr for chronology development 
res = subprocess.call(
 ["/usr/local/bin/Rscript","/Users/julieedwards/Documents/Projects/QWAcode/3_SummarizeAnnualsStats_1.5.R",Workdir2,Workdir3]) 


# Fill gaps and save to WorkDir3
SeriesFiles = []
files = os.listdir(Workdir3)

for file_name in files:
    for i in range(len(Series)):
        for j in range(len(resolution.split(sep=','))):
          if fnmatch.fnmatch(file_name, f"{Series[i]}*{method}*{resolution.split(sep=',')[j]}*"):
            SeriesFiles.append(file_name)
print(set(SeriesFiles))
filled=[]
for filename in SeriesFiles:
  df = pd.read_csv(os.path.join(Workdir3,filename), index_col='YEAR', header=0,sep=' ')
  filled=df.interpolate(method='linear',limit_area='inside')
  filled.to_csv(os.path.join(Workdir3,f"gap_{filename}"))


#reading gap-filled data to dplR to do plotting and detrending :), Change this script to your liking
res = subprocess.call(
  ["/usr/local/bin/Rscript","/Users/julieedwards/Documents/Projects/QWAcode/5_BuildChronology_1.0.R",Workdir3,Workdir4]) 

