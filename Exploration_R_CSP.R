setwd("~/BCfiles/KaggleEEG")

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



#==================================================================
#filtering only 2 subjects 
df_csp_12= filter(df_csp, subject == '1' | subject =='2')
saveRDS(df_csp_12,'df_csp_12.RDS')

#==================================================================
# Select rows that only contain one class
library(dplyr)
#changing to numeric to sum total of classes
df_csp[,6:11]=  sapply(df_csp[,6:11], function (x) as.numeric(as.character(x)))

df_csp['totalcol'] = apply(df_csp[,6:11],1,function(x) sum(x))
df_csp = tbl_df(df_csp)

subset_csp = filter(df_csp,totalcol < 2)
saveRDS(subset_csp,'df_csp_12_solo.RDS')

subset_csp$HandStart=  as.factor(subset_csp$HandStart)

#==================================================================
#Random Forest

library(stats)
library(VIM)
library(AUCRF)
library(rpart)
library(dplyr)

subset_csp = readRDS('df_csp_12_solo.RDS')
subset_csp$HandStart=  as.factor(subset_csp$HandStart)
subset_csp = filter(subset_csp,subject == '1')


set.seed(12345) 
fit = AUCRF(HandStart ~ .,
            ntree = 10,
            data = subset_csp[,c(2,3,4,5,6)]) 

saveRDS(fit,"Models/fit_10t_sub1_solo.rds")

fit = readRDS("Models/fit.rds")
summary(fit)
plot(fit)


#Using cross-validation and random forest:

fit.cv = AUCRFcv(fit,
                 nCV = 10, ###Number of folds.
                 M = 10) ###Number of CV repetitions.
saveRDS(fit.cv,"Models/fit.cv.rds")



fit.cv = readRDS("Models/fit.cv.rds")
summary(fit.cv)
plot(fit.cv)
names(fit.cv)


#==========================================================
#importing 10subjects of solo file 


df_csp_solo = read.csv('csp_solo.csv')
df_csp_solo[,6:11] = as.data.frame( lapply(df_csp_solo[,6:11],
                           function(x) as.factor(as.character(x))))

saveRDS(df_csp_solo,"df_csp_solo.rds")
df_csp_solo = readRDS("df_csp_solo.rds")

# Preparing dataframe to use in pyspark Mlib
library(compositions)

df_csp_solo['class_all'] = unbinary(paste0(df_csp_solo$HandStart,
                                  df_csp_solo$FirstDigitTouch,
                                  df_csp_solo$BothStartLoadPhase,
                                  df_csp_solo$LiftOff,
                                  df_csp_solo$Replace,
                                  df_csp_solo$BothReleased,
                                  sep=''))
df_for_tree = df_csp_solo[,c(12,2,3,4,5)]
write.csv(df_for_tree,'df_for_tree.csv', row.names = FALSE)

#sampling to get a testing dataframe to be used in pyspark
df_for_tree_small = df_for_tree[sample(nrow(df_for_tree), 10000), ]
write.csv(df_for_tree_small,'df_for_tree_small.csv', row.names = FALSE)


#============================================================
#Testing with whole dataset csp_solo with only binary trees
library(caret)

fitControl <- trainControl(## 10-fold CV
      method = "repeatedcv",
      number = 5,
      ## repeated ten times
      repeats = 5)

set.seed(825)
gbmFit1 <- train(HandStart ~ ., data = df_csp_solo[,c(2,3,4,5,6)],
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)






