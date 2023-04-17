# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:light
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: R [conda env:TSIE] *
#     language: R
#     name: conda-env-TSIE-r
# ---

# ## Calculate time-series meta-features

.libPaths()

PATH <- "../data/"

options(repr.matrix.max.cols = 50)

library(data.table)
library(feasts)
library(lubridate)
library(tsibble)
library(moments)
library(dplyr)

# +
#The following functions were taken from https://github.com/pridiltal/oddstream/blob/master/R/extract_tsfeatures.R 
#(accessed 20.04.2020 at 12:38 PM) in a slightly adapted form:

min_max_feat <- function(x){
    c(min = min(x, na.rm = TRUE), max = max(x, na.rm = TRUE))
}

burstFF <- function(x){
    B <- var(x, na.rm = TRUE) / mean(x, na.rm = TRUE)
    c(burst = B)
}

rmeaniqmean <- function(x){
    #Calculation of IQM according to hctsa: https://github.com/benfulcher/hctsa/blob/master/Operations/DN_Mean.m
    quan <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
    R_iqm <- mean(x[x >= quan[1] & x <= quan[2]], na.rm = TRUE) / mean(x, na.rm = TRUE)
    c(ratio_mean_iqmean = R_iqm)
}

dmoments <- function(x){
    momentssd <- moment(x, order = 3, na.rm = TRUE)
    dm <- momentssd / sd(x, na.rm = TRUE)
    c(moment = dm)
}

highlowmu <- function(x){
    mu <- mean(x, na.rm = TRUE)
    mhi <- mean(x[x > mu], na.rm = TRUE)
    mlo <- mean(x[x < mu], na.rm = TRUE)
    hlm <- (mhi - mu) / (mu - mlo)
    c(ratio_high_low_mean = hlm)
}
# -

dt <- fread(paste0(PATH, "timeSeries.csv"))
dt[, TimeStamp := ymd_hms(dt$TimeStamp)]
tbl_ts <- as_tibble(dt) %>% as_tsibble(index = TimeStamp, key = c("PotNumber", "Sensor"))

head(tbl_ts)

#Convert to Kelvin before calculating meta-features
tbl_ts$Value <- tbl_ts$Value + 273.15

head(tbl_ts)

# ## Calculate meta-features

# +
featSet1 <- list("mean" = ~mean(., na.rm = TRUE), 
                 "var" = ~var(., na.rm = TRUE),
                 min_max_feat,
                 rmeaniqmean)

featSet2 <- list(feat_acf,
                 feat_stl,
                 longest_flat_spot,
                 shift_level_max,
                 shift_var_max,
                 var_tiled_mean,
                 var_tiled_var,
                 burstFF,
                 dmoments,
                 highlowmu,
                 feat_spectral)
# -

tbl_feats <- cbind(tbl_ts %>% features(Value, featSet1, .period = 60),
                   tbl_ts %>% features(Value, featSet2, .period = 60) %>% select(-PotNumber, -Sensor))
tbl_feats

tbl_feats %>% 
    select(-acf10, 
           -diff1_acf1, 
           -diff1_acf10, 
           -diff2_acf1, 
           -diff2_acf10, 
           -season_acf1, 
           -stl_e_acf1, 
           -stl_e_acf10, 
           -seasonal_strength_60, 
           -seasonal_peak_60, 
           -seasonal_trough_60, 
           -trend_strength) %>% 
    write.csv(paste0(PATH, "metaFeatures.csv"), row.names = F)
