library(dplR) #import for chronology building

rm(list=ls())

####LCP version

#Calling arguments if R script is called from python
args = commandArgs(trailingOnly=TRUE)
Workdir3= ######## Location of LCL or LCH gap_ files
Workdir4= ###### Location to save LCL and LCH chronologies


#output

setwd(Workdir3)

Files=startsWith(list.files(Workdir3),'gap') #find files that have been  gap-filled
filelist=list.files(Workdir3)[Files] #make list of files to iterate through

exclude=read.csv("_YearsToExclude.txt",sep="\t")

for(i in 1:length(filelist)){

  series=read.rwl(filelist[i])
  write.csv(summary(series),paste(Workdir4,"RWL_summary_",filelist[i],sep='')) # write summary stats of series
  
  
  trunc30=detrend(series[rownames(series)>1916,],
                method = c("Spline"), nyrs = 30,f = 0.5, pos.slope = FALSE,difference=FALSE) #Detrend with a30yr spline 
  Badremoved=trunc30
   for (j in 1:nrow(exclude)) {
    woodid <- exclude$WOODID[j]
    year <- as.character(exclude$YEAR[j])
    
    # Check if the current 'WOODID' and 'year' exist in 'series' DataFrame
    if (woodid %in% colnames(trunc30) && year %in% rownames(trunc30)) {
      # Set the matching cell value to NaN
      Badremoved[year, woodid] <- NaN
      }
    }

  write.csv(rwi.stats(Badremoved,period = 'common'),paste(Workdir4,"RWI_stats",filelist[i],sep=''))
  
  truncChrono=chron(Badremoved,biweight = TRUE)
  write.csv(truncChrono,paste(Workdir4,"Spline30yr_",filelist[i],sep=''))
  

}

