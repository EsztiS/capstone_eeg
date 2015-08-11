library(dplyr)
library(ggplot2)
library(stringr)

#=========== Loading and creating dataframe to be saved in RDS ===============

colClasses=c('numeric', rep('factor', 37), 'character')
#Loading csv
df_csp = read.csv('csp_train.csv')
                  , 
#                  colClasses = c('factor',rep('numeric',4),rep('factor',6)))
df_csp$HandStart = as.factor(df_csp$HandStart)
df_csp$FirstDigitTouch = as.factor(df_csp$FirstDigitTouch)
df_csp$BothStartLoadPhase = as.factor(df_csp$BothStartLoadPhase)
df_csp$LiftOff = as.factor(df_csp$LiftOff)
df_csp$Replace = as.factor(df_csp$Replace)
df_csp$BothReleased  = as.factor(df_csp$BothReleased )

#Creating subject, series and frame columns
df_csp$subject = as.factor(str_extract(df_csp$id,"[0-9]+"))
df_csp$series =as.factor(str_extract(df_csp$id,'(?<=s)([0-9])'))
df_csp$frame = as.factor(str_extract(df_csp$id,'(?<=_)([0-9]+)'))


#Converting to RDS
saveRDS(df_csp,'data_csp.rds')

#==============================Data visualization and exploration ===========================================
#Start here
#Loading RDS

df_csp = readRDS('data_csp.rds')


head(df_csp)
dim(df_csp)

df_csp = tbl_df(df_csp)
#To plot data from one subject:
sub1 = filter( df_csp, subject == 1 ) 
qplot(F1,F2,colour = HandStart,data=sub1)
