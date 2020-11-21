library(data.table)
library(lubridate)

#------------------------------------ 
# merge CEF returns data and masterfile 

data <- as.data.table(read.csv("USCEF_DailyReturnHistory.csv", stringsAsFactors = F))
data$date <- ymd(data$date)

master <- as.data.table(read.csv("master_file.csv", stringsAsFactors = F))

USA <- master[geolevel1=="USA"]
USA[,duplicate:=.N>1, by=ticker]
USA[, match:=sum(marketstatus=="ACTV"), by=ticker]
USA <- USA[!(duplicate==T & match>=1 & marketstatus!="ACTV"),]

USA[,duplicate:=.N>1, by=ticker]
USA <- USA[duplicate!=T,]
USA$duplicate <- NULL
USA$match <- NULL

USA_ticker <- unique(USA$ticker)

data_USA <- data[ticker%in%USA_ticker,]

merged <- merge(data_USA, USA, by=("ticker"), all.x = T)

#------------------------------------ 
# regress discount change on lag expense ratio 
library(lfe)
library(stargazer)

regress_data <- merged[,.(ticker, date, pd, priceclose, nav, expenseratio, assetclasslevel1, assetclasslevel2, assetclasslevel3, inceptiondate, daily_rtn)]

regress_data[, lag_pd:=shift(pd, n=1, type="lag"), by="ticker"]
regress_data[, delta_pd:=(pd/lag_pd - 1), by="ticker"]

regress_data[, abs_pd:=priceclose-nav,]
regress_data[, lag_abs_pd:=shift(abs_pd, n=1, type="lag"), by="ticker"]
regress_data[, delta_abs_pd:=(abs_pd/lag_abs_pd - 1), by="ticker"]


regress_data[, lag_expense:=shift(expenseratio, n=1, type="lag"), by="ticker"]



regress_data[, year:=year(date)]

regress_data <- regress_data[!is.na(lag_expense),]
regress_data <- regress_data[!is.na(delta_pd),]
regress_data <- regress_data[!is.na(lag_rtn),]
regress_data <- regress_data[delta_pd!=Inf & delta_pd!=-Inf,]

panel1 = felm(pd ~ lag_expense , regress_data) 
panel2 = felm(pd ~ lag_expense | 0 | 0 | year, regress_data) 
panel3 = felm(pd ~ lag_expense | 0 | 0 | year + ticker, regress_data) 
panel4 = felm(pd ~ lag_expense | assetclasslevel1 | 0 | year + ticker, regress_data) 
panel5 = felm(pd ~ lag_expense | assetclasslevel2 | 0 | year + ticker, regress_data) 
panel6 = felm(pd ~ lag_expense | assetclasslevel3 | 0 | year + ticker, regress_data) 
panel7 = felm(pd ~ lag_expense | assetclasslevel3 + year | 0 | year + ticker, regress_data)

stargazer(panel1, panel2, panel3, panel4, panel5, panel6, panel7, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats

panel8 = felm(abs_pd ~ lag_expense , regress_data) 
panel9 = felm(abs_pd ~ lag_expense | 0 | 0 | year, regress_data) 
panel10 = felm(abs_pd ~ lag_expense | 0 | 0 | year + ticker, regress_data) 
panel11 = felm(abs_pd ~ lag_expense | assetclasslevel1 | 0 | year + ticker, regress_data) 
panel12 = felm(abs_pd ~ lag_expense | assetclasslevel2 | 0 | year + ticker, regress_data) 
panel13 = felm(abs_pd ~ lag_expense | assetclasslevel3 | 0 | year + ticker, regress_data) 
panel14 = felm(abs_pd ~ lag_expense | assetclasslevel3 + year | 0 | year + ticker, regress_data)

stargazer(panel8, panel9, panel10, panel11, panel12, panel13, panel14, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats

panel15 = felm(delta_abs_pd ~ lag_expense , regress_data) 
panel16 = felm(delta_abs_pd ~ lag_expense | 0 | 0 | year, regress_data) 
panel17 = felm(delta_abs_pd ~ lag_expense | 0 | 0 | year + ticker, regress_data) 
panel18 = felm(delta_abs_pd ~ lag_expense | assetclasslevel1 | 0 | year + ticker, regress_data) 
panel19 = felm(delta_abs_pd ~ lag_expense | assetclasslevel2 | 0 | year + ticker, regress_data) 
panel20 = felm(delta_abs_pd ~ lag_expense | assetclasslevel3 | 0 | year + ticker, regress_data) 
panel21 = felm(delta_abs_pd ~ lag_expense | assetclasslevel3 + year | 0 | year + ticker, regress_data)

stargazer(panel15, panel16, panel17, panel18, panel19, panel20, panel21, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats

# all regressions gave non-significant coefficient with low R2 (specially for delta pd) as the expense ratio is annual data and pd/delta_pd is daily data


#------------------------------------ 
# subset only last data point for each year
regress_data_yr <- regress_data[, list(lag_expense=last(lag_expense), 
                                       pd=last(pd), 
                                       abs_pd=last(abs_pd), 
                                       delta_pd=last(delta_pd), 
                                       delta_abs_pd=last(delta_abs_pd), 
                                       assetclasslevel1=last(assetclasslevel1),
                                       assetclasslevel2=last(assetclasslevel2),
                                       assetclasslevel3=last(assetclasslevel3)), 
                                by=c("ticker", "year")]

panel22 = felm(pd ~ lag_expense , regress_data_yr) 
panel23 = felm(pd ~ lag_expense | 0 | 0 | year, regress_data_yr) 
panel24 = felm(pd ~ lag_expense | 0 | 0 | year + ticker, regress_data_yr) 
panel25 = felm(pd ~ lag_expense | assetclasslevel1 | 0 | year + ticker, regress_data_yr) 
panel26 = felm(pd ~ lag_expense | assetclasslevel2 | 0 | year + ticker, regress_data_yr) 
panel27 = felm(pd ~ lag_expense | assetclasslevel3 | 0 | year + ticker, regress_data_yr) 
panel28 = felm(pd ~ lag_expense | assetclasslevel3 + year | 0 | year + ticker, regress_data_yr)
panel29 = felm(pd ~ lag_expense | assetclasslevel2 + year | 0 | year + ticker, regress_data_yr) 

stargazer(panel22, panel23, panel24, panel25, panel26, panel27, panel28, panel29, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats
# still non significant coefficient

#------------------------------------ 
# regress discount change on lag expense ratio and age of CEF
library(lfe)
library(stargazer)

regress_data <- merged[,.(ticker, date, pd, priceclose, nav, expenseratio, assetclasslevel1, assetclasslevel2, assetclasslevel3, inceptiondate, daily_rtn, volume)]

regress_data[, lag_pd:=shift(pd, n=1, type="lag"), by="ticker"]
regress_data[, delta_pd:=(pd/lag_pd - 1), by="ticker"]

regress_data[, abs_pd:=priceclose-nav,]
regress_data[, lag_abs_pd:=shift(abs_pd, n=1, type="lag"), by="ticker"]
regress_data[, delta_abs_pd:=(abs_pd/lag_abs_pd - 1), by="ticker"]

regress_data[, lag_expense:=shift(expenseratio, n=1, type="lag"), by="ticker"]

regress_data[, lag_rtn:=shift(daily_rtn, n=1, type="lag"), by="ticker"]

regress_data[, cd:=pd-lag_pd, by="ticker"]
regress_data[, cd_lag1:=shift(cd, n=1, type="lag"), by="ticker"]
regress_data[, cd_lag2:=shift(cd, n=2, type="lag"), by="ticker"]
regress_data[, cd_lag3:=shift(cd, n=3, type="lag"), by="ticker"]
regress_data[, cd_lag4:=shift(cd, n=4, type="lag"), by="ticker"]
regress_data[, cd_lag5:=shift(cd, n=5, type="lag"), by="ticker"]
regress_data[, cd_lag6:=shift(cd, n=6, type="lag"), by="ticker"]

regress_data[, year:=year(date)]

# create age variable
regress_data$inceptiondate <- mdy(regress_data$inceptiondate)
regress_data[,age:=log(as.integer(date-inceptiondate))]

# add price of CEF
regress_data[, lag_price:=shift(priceclose, n=1, type="lag"), by="ticker"]

regress_data <- regress_data[!is.na(lag_expense),]
regress_data <- regress_data[!is.na(delta_pd),]
regress_data <- regress_data[!is.na(age),]
regress_data <- regress_data[!is.na(lag_price),]
regress_data <- regress_data[!is.na(lag_rtn),]
regress_data <- regress_data[delta_pd!=Inf & delta_pd!=-Inf,]

panel30 = felm(pd ~ lag_expense + age , regress_data) 
panel31 = felm(pd ~ lag_expense + age| 0 | 0 | year, regress_data) 
panel32 = felm(pd ~ lag_expense + age| 0 | 0 | year + ticker, regress_data) 
panel33 = felm(pd ~ lag_expense + age| assetclasslevel1 | 0 | year + ticker, regress_data) 
panel34 = felm(pd ~ lag_expense + age| assetclasslevel2 | 0 | year + ticker, regress_data) 
panel35 = felm(pd ~ lag_expense + age| assetclasslevel3 | 0 | year + ticker, regress_data) 
panel36 = felm(pd ~ lag_expense + age| assetclasslevel3 + year | 0 | year + ticker, regress_data)

stargazer(panel30, panel31, panel32, panel33, panel34, panel35, panel36, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats
# age is significant predictor but expenses are not. sign on expenses is wrong


# subset only last data point for each year
regress_data_yr <- regress_data[, list(lag_expense=last(lag_expense), 
                                       pd=last(pd), 
                                       abs_pd=last(abs_pd), 
                                       delta_pd=last(delta_pd), 
                                       delta_abs_pd=last(delta_abs_pd), 
                                       assetclasslevel1=last(assetclasslevel1),
                                       assetclasslevel2=last(assetclasslevel2),
                                       assetclasslevel3=last(assetclasslevel3), 
                                       age=last(age)), 
                                by=c("ticker", "year")]

panel37 = felm(pd ~ lag_expense + age , regress_data_yr) 
panel38 = felm(pd ~ lag_expense + age | 0 | 0 | year, regress_data_yr) 
panel39 = felm(pd ~ lag_expense + age | 0 | 0 | year + ticker, regress_data_yr) 
panel40 = felm(pd ~ lag_expense + age | assetclasslevel1 | 0 | year + ticker, regress_data_yr) 
panel41 = felm(pd ~ lag_expense + age | assetclasslevel2 | 0 | year + ticker, regress_data_yr) 
panel42 = felm(pd ~ lag_expense + age | assetclasslevel3 | 0 | year + ticker, regress_data_yr) 
panel43 = felm(pd ~ lag_expense + age | assetclasslevel3 + year | 0 | year + ticker, regress_data_yr)
panel44 = felm(pd ~ lag_expense + age | assetclasslevel2 + year | 0 | year + ticker, regress_data_yr) 

stargazer(panel37, panel38, panel39, panel40, panel41, panel42, panel43, panel44, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats
# age is significant predictor but expenses still are not. sign is correct now for expenses.

#------------------------------------
# try on delta_pd

panel45 = felm(delta_abs_pd ~ lag_expense + age | assetclasslevel2 | 0 | year + ticker, regress_data) 
panel46 = felm(delta_abs_pd ~ lag_expense + age | assetclasslevel3 | 0 | year + ticker, regress_data) 
panel47 = felm(delta_abs_pd ~ lag_expense + age | assetclasslevel3 + year | 0 | year + ticker, regress_data)

stargazer(panel45, panel46, panel47, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats
# non significant and low R2

#------------------------------------
# add price of CEF

panel48 = felm(delta_abs_pd ~ lag_expense + age + lag_price| assetclasslevel2 | 0 | year + ticker, regress_data) 
panel49 = felm(delta_abs_pd ~ lag_expense + age + lag_price| assetclasslevel3 | 0 | year + ticker, regress_data) 
panel50 = felm(delta_abs_pd ~ lag_expense + age + lag_price| assetclasslevel3 + year | 0 | year + ticker, regress_data)

stargazer(panel48, panel49, panel50, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats

#------------------------------------
# add lag return of CEF

panel51 = felm(delta_abs_pd ~ age + lag_rtn | assetclasslevel2 | 0 | year + ticker, regress_data) 
panel52 = felm(delta_abs_pd ~ age + lag_rtn | assetclasslevel3 | 0 | year + ticker, regress_data) 
panel53 = felm(delta_abs_pd ~ age + lag_rtn | assetclasslevel3 + year | 0 | year + ticker, regress_data)

stargazer(panel51, panel52, panel53, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats

#------------------------------------
unique(regress_data$year)

test <- regress_data[year%in%c(2018, 2019, 2020),]

panel54 = felm(delta_abs_pd ~ age + lag_rtn | assetclasslevel2 | 0 | year + ticker, test) 
panel55 = felm(delta_abs_pd ~ age + lag_rtn | assetclasslevel3 | 0 | year + ticker, test) 
panel56 = felm(delta_abs_pd ~ age + lag_rtn | assetclasslevel3 + year | 0 | year + ticker, test)

panel54 = felm(delta_abs_pd ~ lag_rtn | assetclasslevel2 | 0 | year + ticker, test) 
panel55 = felm(delta_abs_pd ~ lag_rtn | assetclasslevel3 | 0 | year + ticker, test) 
panel56 = felm(delta_abs_pd ~ lag_rtn | assetclasslevel3 + year | 0 | year + ticker, test)

stargazer(panel54, panel55, panel56, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats

#------------------------------------
# regress cd on lag cd

regress_data[, cd:=pd-lag_pd, by="ticker"]
regress_data[, cd_lag1:=shift(cd, n=1, type="lag"), by="ticker"]
regress_data[, cd_lag2:=shift(cd, n=2, type="lag"), by="ticker"]
regress_data[, cd_lag3:=shift(cd, n=3, type="lag"), by="ticker"]
regress_data[, cd_lag4:=shift(cd, n=4, type="lag"), by="ticker"]
regress_data[, cd_lag5:=shift(cd, n=5, type="lag"), by="ticker"]
regress_data[, cd_lag6:=shift(cd, n=6, type="lag"), by="ticker"]

regress_data[, year:=year(date)]

test <- regress_data[year%in%c(2018, 2019, 2020),]
test[, volume:=volume/1000000,]

panel57 = felm(cd ~ cd_lag1 + cd_lag2 + cd_lag3 + cd_lag4 + cd_lag5 + cd_lag6 | assetclasslevel2 | 0 | year + ticker, test) 
panel58 = felm(cd ~ cd_lag1 + cd_lag2 + cd_lag3 + cd_lag4 + cd_lag5 + cd_lag6 + lag_pd | assetclasslevel2 | 0 | year + ticker, test) 
panel59 = felm(cd ~ cd_lag1 + cd_lag2 + cd_lag3 + cd_lag4 + cd_lag5 + cd_lag6 + lag_pd + volume | assetclasslevel2 | 0 | year + ticker, test) 

panel60 = felm(cd ~ cd_lag1 + lag_pd + volume + age | assetclasslevel1 | 0 | year + ticker, test) 
panel61 = felm(cd ~ cd_lag1 + lag_pd + volume + age | assetclasslevel2 | 0 | year + ticker, test) 
panel62 = felm(cd ~ cd_lag1 + lag_pd + volume | assetclasslevel3 | 0 | year + ticker, test) 
panel63 = felm(cd ~ cd_lag1 + lag_pd + volume | assetclasslevel3 + year | 0 | year + ticker, test) 


stargazer(panel57, panel58, panel59, panel60, panel61, panel62, panel63,type = 'text', report = 'vc*st') # Output regressions as text, report t-stats


panel64 = felm(cd ~ cd_lag1 + lag_pd + volume + age | assetclasslevel3 | 0 | year + ticker, test) 

stargazer(panel60, panel61,panel64, type = 'text', report = 'vc*st') # Output regressions as text, report t-stats

cef <- unique(regress_data$ticker)

acf(regress_data[ticker==cef[19], cd])




#------------------------------------
regress_data

summary(regress_data$expenseratio)
summary(regress_data$delta_pd)
sum(regress_data$delta_pd==Inf)

summary(regress_data$delta_abs_pd)

write.csv(merged, file="US_data.csv")

merged[, lag_pd_1:=shift(pd, 1, "lag"), by="ticker"]
merged[, ave_pd:=mean(pd, na.rm=T), by="ticker"]

sum(is.na(merged$pd))
sum(is.na(merged$expenseratio))
summary(merged$expenseratio)


lag_pd_1 <- merged[,.(date, ticker, pd)]
lag_pd_1[,lag1:=date+1,]

test <- merge(merged, lag_pd_1[,.(lag1, ticker, pd)], by.x=c("date", "ticker"), by.y=c("lag1", "ticker"))


