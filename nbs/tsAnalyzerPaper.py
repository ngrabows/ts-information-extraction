# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:TSIE] *
#     language: python
#     name: conda-env-TSIE-py
# ---

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# +
def is_space(point_, coords_):
    coords_ = np.array(coords_)
    return np.all((point_[0] < coords_[:,0]) | 
                  (point_[0] > coords_[:,1]) | 
                  (point_[1] < coords_[:,2]) | 
                  (point_[1] > coords_[:,3]))

def check_all(point_, size_, coords_):
    return np.all([is_space(point_, coords_), 
            is_space([point_[0] + size_, point_[1]], coords_), 
            is_space([point_[0] + size_, point_[1] + size_], coords_), 
            is_space([point_[0], point_[1] + size_], coords_)])

def standard_scaler(df):
    for c,s in df.items():
        if (is_numeric_dtype(s)) and (c != "PotNumber"):
            df[c] = (s - s.mean())/s.std(ddof = 0)
    return df

def create_pca_plot(df, pca, df_ts, size_ = 1, 
                    figsize = (12,10), ticks_labelsize = 12, plot_ts = True, 
                    ts_title_fontsize = 8, label_fontsize = 14, xlim = None, 
                    ylim = None, title_ = None, titlepad = 0, plotTitles = [], excludeTSplot = [], plotTS = True):    
    
    fig, ax = plt.subplots(1,1, figsize = figsize)
    coords = []
    for s in df["Sensor"].unique():
        X_ = df[df["Sensor"] == s][["PCA1", "PCA2", "PotNumber"]]
        ax.scatter(X_["PCA1"], X_["PCA2"], alpha=1, s = 96, color = "#89ba17", zorder = 1)
        if plot_ts:    
            pots_ = X_["PotNumber"].unique()
            for i, pot in enumerate(pots_):
                point_ = X_[["PCA1", "PCA2"]].iloc[i]
                sensorPotName = f"{s}-{pot}"
                if ((np.size(coords) == 0) or check_all(point_, size_, coords)):

                    coords.append([point_[0], point_[0] + size_, point_[1], point_[1] + size_])
                    if sensorPotName not in excludeTSplot:
                        ax.scatter(point_[0], point_[1], facecolors = 'none', edgecolors = 'black', s = 96*2, zorder = 2)
                        if plotTS:
                            ax1 = ax.inset_axes((point_[0] + size_/5, point_[1] - size_/2, size_, size_), transform=ax.transData)
                            ts = df_ts[(df_ts["PotNumber"] == pot) & (df_ts["Sensor"] == s)]
                            ax1.plot(ts.index, ts["Value"], color = 'black', linewidth = 1)

                            ax1.spines["bottom"].set_visible(False)
                            ax1.spines["top"].set_visible(False)
                            ax1.spines["left"].set_visible(False)
                            ax1.spines["right"].set_visible(False)
                            ax1.xaxis.set_visible(False)
                            ax1.yaxis.set_visible(False)
                            ax1.patch.set_visible(False)
                            if (len(plotTitles) == 0) | (sensorPotName in plotTitles):                        
                                ax1.set_title(sensorPotName, fontsize = ts_title_fontsize, pad = titlepad)

    if title_ is not None:
        ax.set_title(title_, fontsize = ts_title_fontsize)
    ax.set_xlabel("Principal Component 1 (explained variance " + str(np.round(pca.explained_variance_ratio_[0]*100,1)) + "%)", fontsize= label_fontsize)
    ax.set_ylabel("Principal Component 2 (explained variance " + str(np.round(pca.explained_variance_ratio_[1]*100,1)) + "%)", fontsize= label_fontsize)
    ax.tick_params(labelsize = ticks_labelsize)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid()


# -

# ## Parameter

sensors = ["TCRL1", "TCRE1", "TCRL2", "TCRL3", 
           "TCWT1", "TCEW1", "TCWT2", "TCEW2", "TCWT3", 
           "TCUW1", "TCUW2", "TCOR1", "TCAG1", 
           "TCWT6", "TCEW4", "TCWT5", "TCEW3", "TCWT4", 
           "TCRL6", "TCRL5", "TCAL1", "TCRL4"]

PATH = "../data/"
PATH_SAVE_PLOTS = "../plots/"

start = "2020-10-08 07:00"
end = "2020-10-09 07:00"
potStart = 1001
potEnd = 1120
resample_resolution = "1min"

start_1h = (pd.to_datetime(start) - pd.Timedelta("1h")).strftime("%Y-%m-%d %H:%M:%S")
end_1h = (pd.to_datetime(end) + pd.Timedelta("1h")).strftime("%Y-%m-%d %H:%M:%S")

# ## Load raw sensor data set

df = pd.read_csv(f"{PATH}/timeSeriesRaw.csv", parse_dates=["TimeStamp"])

# +
df["PotNumber"] = df["PotNumber"] + 1000
df.set_index("TimeStamp", inplace = True)
df["PotNumber"] = df["PotNumber"].astype(np.uint16)

pots = np.sort(df["PotNumber"].unique())
df_pots = []

for pot in pots:
    df_pot = df[df["PotNumber"] == pot].copy()
    df_sensors = []
    for sensor in sensors:
        df_sensor = df_pot[df_pot["Sensor"] == sensor].copy()
        df_sensor = pd.DataFrame(df_sensor["Value"].resample(resample_resolution, label='right', closed='right').mean())
        df_sensor["PotNumber"] = pot
        df_sensor["Sensor"] = sensor
        df_sensors.append(df_sensor)
    df_pots.append(pd.concat(df_sensors))
df_ts = pd.concat(df_pots)
# -

df_ts.head()

# Check for missing values

df_ts[df_ts["Value"].isna()]

# +
len_pots = len(df_ts["PotNumber"].unique())
len_sensors = len(df_ts["Sensor"].unique())

print(f"{len_pots} cells are considered. There are {len_sensors} thermocouples mounted on each cell.")
print(f"This results in a total of {len_sensors*len_pots } time series.")
# -

# Export time series for calculating time series meta-features in R.
df_ts.to_csv(f"{PATH}/timeSeries.csv", index = True)

# Load time series data if not already done.
df_ts = pd.read_csv(f"{PATH}/timeSeries.csv")

# ## Load and Preprocess time series meta-features
# **Run the R script ``calcTSMetaFeatures.R`` to get the time series meta-features.**

df_feats = pd.read_csv(f"{PATH}/metaFeatures.csv")
df_feats = df_feats.replace([-np.inf, np.inf], np.nan)

#Show features
print(f"Number of meta-features: {len(df_feats.columns[2:])}")
for f in df_feats.columns[2:]: print(f)


df_feats.describe(include = "all")

df_feats_nan = df_feats[df_feats.isna().any(axis=1)]
print(f"For {len(df_feats_nan)} time series, at least one time series meta-feature was calculated that is NaN. These time series were removed." )

# ## PCA

df_feats = df_feats.dropna(axis=0)
ids_ = df_feats["Sensor"] + "-" + df_feats["PotNumber"].astype(str)

df_feats.head()

# +
df_feats_std = standard_scaler(df_feats.copy())

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(df_feats_std.iloc[:, 2:])
# -

df_feats_augmented = df_feats_std.assign(PCA1 = X_transformed[:,0], 
                                         PCA2 = X_transformed[:,1])

plot_titles = ["TCRL2-1012", "TCWT2-1068", "TCWT3-1118", "TCWT2-1118", "TCWT6-1118", "TCWT5-1118", "TCWT1-1056", "TCRL3-1061"]

create_pca_plot(df_feats_augmented, 
                pca, 
                df_ts, 
                size_ = 3,#4.5 
                ticks_labelsize = 20, 
                ts_title_fontsize= 18,
                label_fontsize = 22,
                figsize = (10,8),
                xlim = [-5, 35],
                ylim = [-20, 25],
                titlepad = 2,
                plotTS = True,
                plotTitles = plot_titles)
plt.savefig(f"{PATH_SAVE_PLOTS}grabo3.pdf")

ts_weird = [
    "TCRL2-1012",
    "TCRL5-1026",
    "TCWT2-1056",
    "TCRL3-1061",
    "TCRL6-1061",
    "TCWT2-1068",
    "TCRL1-1084",
    "TCWT2-1118",
    "TCWT3-1118",
    "TCWT6-1118",
    "TCWT5-1118",
    "TCWT4-1061",
    "TCWT5-1013"]

df_feats = df_feats[~ids_.isin(ts_weird)]
ids_ = df_feats["Sensor"] + "-" + df_feats["PotNumber"].astype(str)

plot_titles = ["TCWT5-1012",
               "TCWT4-1033", 
               "TCWT6-1116", 
               "TCWT6-1012", 
               "TCWT2-1115", 
               "TCWT1-1056", 
               "TCRL2-1056", 
               "TCWT3-1001", 
               "TCWT4-1111", 
               "TCWT1-1115", 
               "TCWT2-1116"]

# +
df_feats_std = standard_scaler(df_feats.copy())

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(df_feats_std.iloc[:, 2:])

df_feats_augmented = df_feats_std.assign(PCA1 = X_transformed[:,0], 
                                     PCA2 = X_transformed[:,1])


create_pca_plot(df_feats_augmented, 
                pca, 
                df_ts, 
                size_ = 2, 
                ticks_labelsize = 20, 
                ts_title_fontsize= 18,
                label_fontsize = 22,
                figsize = (10,8),
                xlim = [-7.5,10],
                ylim = [-4,40],
                titlepad = 2,
                plotTitles = plot_titles)
plt.savefig(f"{PATH_SAVE_PLOTS}grabo4.pdf")
# -

ts_weird = ["TCWT5-1012", 
            "TCWT4-1033",
            "TCWT6-1116"]

df_feats = df_feats[~ids_.isin(ts_weird)]
ids_ = df_feats["Sensor"] + "-" + df_feats["PotNumber"].astype(str)

plot_titles = ["TCWT3-1068", 
               "TCWT2-1115",
               "TCWT3-1061",
               "TCWT3-1001",
               "TCWT6-1116",
               "TCWT1-1115", 
               "TCRL1-1115", 
               "TCWT6-1111", 
               "TCUW1-1012", 
               "TCWT1-1119", 
               "TCWT4-1111", 
               "TCWT6-1012"]

# +
df_feats_std = standard_scaler(df_feats.copy())
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(df_feats_std.iloc[:, 2:])

df_feats_augmented = df_feats_std.assign(PCA1 = X_transformed[:,0], 
                                     PCA2 = X_transformed[:,1])

create_pca_plot(df_feats_augmented, 
                pca, 
                df_ts, 
                size_ = 2,#1.75,#3.5
                ticks_labelsize = 20, 
                ts_title_fontsize= 18,
                label_fontsize = 22,
                figsize = (10,8),
                xlim=[-7.5,20],
                ylim = [-5, 20],
                titlepad = 2,
                plotTitles = plot_titles)
plt.savefig(f"{PATH_SAVE_PLOTS}grabo5.pdf")
