library(dplR) #import for chronology building

rm(list=ls())

####LCP version
####### test chron.stabilized(x, winLength, biweight = TRUE, running.rbar = FALSE)

#Calling arguments if R script is called from python
args = commandArgs(trailingOnly=TRUE)
Workdir3= toString(args[1])
Workdir4= toString(args[2])


#output

setwd(Workdir3)

Files=startsWith(list.files(Workdir3),'gap') #find files that have been  gap-filled
filelist=list.files(Workdir3)[Files] #make list of files to iterate through



for(i in 1:length(filelist)){

  series=read.rwl(filelist[i])
  #write.csv(series,paste(Workdir4,"ARSTAN_",filelist[i],sep=''))#loading in gap_filled summary files 
  write.csv(summary(series),paste(Workdir4,"RWL_summary_",filelist[i],sep='')) # write summary stats of series
  
  detrended=detrend(series,make.plot = FALSE,
                    method = c("Spline"), nyrs=30,f = 0.5, pos.slope = FALSE,difference=FALSE) #Detrend with a 2/3 segment length spline (default length)

  

  chrono=chron(detrended,biweight = TRUE) #create chronology
  
  trunc30=detrend(series[rownames(series)>1916,],
                method = c("Spline"), nyrs = 30,f = 0.5, pos.slope = FALSE,difference=FALSE) #Detrend with a 2/3 segment length spline (default length)
  write.csv(rwi.stats(trunc30,period = 'common'),paste(Workdir4,"RWI_stats",filelist[i],sep=''))
  
  truncChrono=chron(trunc30,biweight = TRUE)
  write.csv(truncChrono,paste(Workdir4,"Spline30yr_",filelist[i],sep=''))
  

}

