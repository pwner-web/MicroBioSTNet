import os
import sys
bin = os.path.abspath(os.path.dirname(__file__))
sys.path.append(bin + "/lib")
import argparse
import pickle as pk
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font = fm.FontProperties(size=5)
import torch
import torch.nn as nn
from scipy import stats
from scipy.stats import sem
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import json
from properscoring import crps_ensemble
from pyecharts.charts import ThemeRiver
from pyecharts import options as opts
from datetime import datetime, timedelta

from stgcn import TwoStreamSTGCN
from stgcn import LSTM


__doc__ = """
MicroBioSTNet 
"""
__author__="Gao Shichen"
__mail__= "gaoshichend@163.com"
__date__= "2024/10/09"
__update__ = "2024/12/24"

colors = [("#D92B03","#FEC0C1","#E7DAD2"),
          ("#088C00","#BEEFBF","#FA7F6F"),
          ("#c72228","#F5867F","#F98F34"),
          ("#0C4E9B","#6B98C4","#6BBC46"),
          ("#A5405E","#F2CDCF","#DB6C76"),
          ("#000000","#C2C2C2","#EEBEC0"),
          ("#2878b5","#9ac9db","#f8ac8c"),
          ("#c82423","#ff8884","#A1A9D0"),
          ("#F0988C","#B883D4","#9E9E9E"),
          ("#CFEAF1","#C4A5DE","#F6CAE5"),
          ("#96CCCB","#F27970","#BB9727"),
          ("#54B345","#32B897","#05B9E2")]

def pseudo_date_list(start_date, end_date):
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y/%m/%d'))
        current_date += timedelta(days=1)
    return date_list

def pseudo_date_list(start_date, end_date):
    delta = end_date - start_date
    return [start_date + timedelta(days=i) for i in range(delta.days + 1)]

def stream_plot(data_dict, categories, time_points, types, subject, name):
    data = []
    families = []
    for key in data_dict:
        families.append(key)
        start_date = datetime(2015, 7, 1)
        end_date = start_date + timedelta(days=len(data_dict[key]))
        date_list = pseudo_date_list(start_date, end_date)
        for i in range(0, len(data_dict[key])):
            data.append([date_list[i].strftime('%Y-%m-%d'), float(abs(data_dict[key][i])), key])
    
    theme_river = (
        ThemeRiver(init_opts=opts.InitOpts(width="1600px", height="800px"))
        .add(
            series_name=families,
            data=data,
            singleaxis_opts=opts.SingleAxisOpts(type_="time", pos_top="50", pos_bottom="20"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Temporal Dynamics of {types} Microbial Abundance in {subject} ({name} Level)".format(name=name, types=types, subject=subject),
                pos_top="1%",  
                pos_left="25%",
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(
                pos_top="7%", 
            ),
        )
    )
    theme_river.render("results/{subject}-{types}-{name}-stream_plot.html".format(subject=subject, types=types, name=name))


def averaged(data, ldata, families, lfamilies, time_points, title, name, types, subject):
    time_points = np.array(time_points)
    for family in families:
        try:
            np.mean(data[family])
        except:
            print("cut off")
            print(family)
            print(data[family])
            pass
    sum_abundance = {family: np.sum(data[family], axis=0) for family in families}
    mean_abundance = {family: np.mean(data[family], axis=0) for family in families}
    sem_abundance = {family: stats.sem(data[family], axis=0) for family in families}

    lsum_abundance = {family: np.sum(ldata[family], axis=0) for family in families}
    lmean_abundance = {family: np.mean(ldata[family], axis=0) for family in families}
    lsem_abundance = {family: stats.sem(ldata[family], axis=0) for family in families}

    if len(families) == 2:
        try:
            if "Saliva" in subject:
                baseline = mean_abundance["Rothia"]
                predictline = mean_abundance["Rothia_predict"]
                with open("checkpoints/curve-MicroBioSTNet-{subject}.pk".format(subject = subject), "wb") as fd:
                    pk.dump((baseline, predictline), fd)
                lbaseline = lmean_abundance["Rothia"]
                lpredictline = lmean_abundance["Rothia_predict"]
                with open("checkpoints/curve-LSTM-{subject}.pk".format(subject = subject), "wb") as fd:
                    pk.dump((lbaseline, lpredictline), fd)
            elif "Stool" in subject:
                baseline = mean_abundance["Bacteroides"]
                predictline = mean_abundance["Bacteroides_predict"]
                with open("checkpoints/curve-MicroBioSTNet-{subject}.pk".format(subject = subject), "wb") as fd:
                    pk.dump((baseline, predictline), fd)
                lbaseline = lmean_abundance["Bacteroides"]
                lpredictline = lmean_abundance["Bacteroides_predict"]
                with open("checkpoints/curve-LSTM-{subject}.pk".format(subject = subject), "wb") as fd:
                    pk.dump((lbaseline, lpredictline), fd)
            else:
                print("Note: continue.")
        except:
            print(subject)
            for key in data:
                print(key)
        sys.exit()

    iters = 0
    fn = len(families)/2
    high = int(fn*2)
    fig, axes = plt.subplots(nrows=int(fn),ncols=1,sharex=True,figsize=(8,high)) 
    fig.suptitle('Abundance of {types} Level Over Time ({subject}) - 8 Steps'.format(types=types,subject=subject))
    boxdata = []
    for family in families:
        if "_predict" in family:
            for value in mean_abundance[family]:
                boxdata.append({'Abundance': value, '{types}'.format(types=types): family.split("_predict")[0],'Dataset': "Predicted (MicroBioSTNet)"})
            for value in lmean_abundance[family]:
                boxdata.append({'Abundance': value, '{types}'.format(types=types): family.split("_predict")[0],'Dataset': "Predicted (LSTM)"})
        else:
            for value in mean_abundance[family]:
                boxdata.append({'Abundance': value, '{types}'.format(types=types): family,'Dataset': "Ground Truth"})
        if iters % 2 == 0:
            print("iters: {0}/{1}".format(int(fn),int(iters/2 +1)))
            means = np.array(mean_abundance[family])
            sems = np.nan_to_num(np.array(sem_abundance[family]), nan=0.0, posinf=0.0, neginf=0.0)
            lmeans = np.array(lmean_abundance[family])
            lsems = np.nan_to_num(np.array(lsem_abundance[family]), nan=0.0, posinf=0.0, neginf=0.0)
            y1_smooth = means
            y2_smooth = sems
            ly1_smooth = lmeans
            ly2_smooth = lsems
            x_smooth = time_points[len(time_points)-len(y1_smooth):]
            if "_predict" in family:
                axes[int(iters/2)].plot(x_smooth, y1_smooth, label=family.replace("_predict", "")+" (MicroBioSTNet)", linewidth=2.0, alpha=0.7, color="#d62728")
                axes[int(iters/2)].plot(x_smooth, ly1_smooth, label=family.replace("_predict", "")+" (LSTM)", linewidth=1.0, alpha=0.7, color="#ff7f0e", linestyle="--")
            else:
                axes[int(iters/2)].plot(x_smooth, y1_smooth, label=family+" (Ground Truth)", linewidth=2.0, alpha=0.7, color="#7c7979")
            if types == "Phylum":
                if "_predict" in family:
                    axes[int(iters/2)].fill_between(x_smooth, y1_smooth - 1 * y2_smooth, y1_smooth + 1 * y2_smooth, alpha=0.1, color='red')
                    axes[int(iters/2)].fill_between(x_smooth, ly1_smooth - 1 * ly2_smooth, ly1_smooth + 1 * ly2_smooth, alpha=0.1, color='yellow')
                else:
                    axes[int(iters/2)].fill_between(x_smooth, y1_smooth - 1 * y2_smooth, y1_smooth + 1 * y2_smooth, alpha=0.1, color='#7f7f7f')
        else:
            means = np.array(mean_abundance[family])
            sems = np.nan_to_num(np.array(sem_abundance[family]), nan=0.0, posinf=0.0, neginf=0.0)
            lmeans = np.array(lmean_abundance[family])
            lsems = np.nan_to_num(np.array(lsem_abundance[family]), nan=0.0, posinf=0.0, neginf=0.0)
            y1_smooth = means
            y2_smooth = sems
            ly1_smooth = lmeans
            ly2_smooth = lsems
            x_smooth = time_points[len(time_points)-len(y1_smooth):]
            if "_predict" in family:
                axes[int(iters/2)].plot(x_smooth, y1_smooth, label=family.replace("_predict", "")+" (MicroBioSTNet)", linewidth=2.0, alpha=0.7, color="#d62728")
                axes[int(iters/2)].plot(x_smooth, ly1_smooth, label=family.replace("_predict", "")+" (LSTM)", linewidth=1.0, alpha=0.7, color="#ff7f0e", linestyle="--")
            else:
                axes[int(iters/2)].plot(x_smooth, y1_smooth, label=family+" (Ground Truth)", linewidth=2.0, alpha=0.7, color="#7c7979")
            if types == "Phylum":
                if "_predict" in family:
                    axes[int(iters/2)].fill_between(x_smooth, y1_smooth - 0.7 * y2_smooth, y1_smooth + 0.7 * y2_smooth, alpha=0.1, color='red')
                    axes[int(iters/2)].fill_between(x_smooth, ly1_smooth - 0.7 * ly2_smooth, ly1_smooth + 0.7 * ly2_smooth, alpha=0.1, color='yellow')
                else:
                    axes[int(iters/2)].fill_between(x_smooth, y1_smooth - 0.7 * y2_smooth, y1_smooth + 0.7 * y2_smooth, alpha=0.1, color='#7f7f7f')
            axes[int(iters/2)].legend()
            if int(fn) == int(iters/2+1):
                axes[int(iters/2)].spines['right'].set_visible(False)
                axes[int(iters/2)].spines['top'].set_visible(False)
                axes[int(iters/2)].tick_params(axis='x', labelbottom=True)
                axes[int(iters/2)].set_xlabel('Date (days)')
                base_tick = int(int(len(y1_smooth)/5)/4)
                ticks_to_display = [base_tick*5, base_tick*10, base_tick*15, base_tick*20]
                axes[int(iters/2)].set_xticks(ticks_to_display)
            elif ((int(fn) // 2) + 1) == int(iters/2+1):
                axes[int(iters/2)].set_ylabel('Abundance')
                axes[int(iters/2)].set_xticks([])
                axes[int(iters/2)].spines['right'].set_visible(False)
                axes[int(iters/2)].spines['top'].set_visible(False)
                axes[int(iters/2)].spines['bottom'].set_visible(False)
            else:
                axes[int(iters/2)].tick_params(axis='x', labelbottom=False)
                axes[int(iters/2)].set_xticks([])
                axes[int(iters/2)].spines['right'].set_visible(False)
                axes[int(iters/2)].spines['top'].set_visible(False)
                axes[int(iters/2)].spines['bottom'].set_visible(False)
        iters += 1
    plt.savefig("results/{subject}-{name}_average_relative_abundance.jpg".format(name=name, subject=subject), dpi=300)
    plt.clf()
    plt.close()

    df = pd.DataFrame(boxdata)
    print(df)
    plt.figure(figsize=(9, 7)) 
    palette = ['#2f7fc1', '#d8383a', '#f3d266']
    fig = sns.violinplot(x = '{types}'.format(types=types), y = 'Abundance', data=df, hue='Dataset', hue_order=['Ground Truth', 'Predicted (MicroBioSTNet)', 'Predicted (LSTM)'], palette=sns.color_palette("hls", 8))
    boxplot = fig.get_figure()
    fig.set_title("{types}-level distribution of ground truth vs. predicted data for {subject} - 8 Steps".format(types=types, subject=subject))
    boxplot.savefig("results/{subject}-{name}_boxplot.jpg".format(name=name, subject=subject), dpi=300)
    plt.clf()
    plt.close()
    return mean_abundance

def averaged_1(data, families, time_points, title, name, types, subject, lstm_type):
    time_points = np.array(time_points)
    for family in families:
        try:
            np.mean(data[family])
        except:
            print("cut off")
            print(family)
            print(data[family])
            pass
    sum_abundance = {family: np.sum(data[family], axis=0) for family in families}
    mean_abundance = {family: np.mean(data[family], axis=0) for family in families}
    sem_abundance = {family: stats.sem(data[family], axis=0) for family in families}

    data_t_tmp = []
    data_t = []
    data_o = []
    columns_t = []
    columns_o = []
    residual = 0
    for family in families:
        if "_predict" in family:
            columns_o.append(family)
            data_o.append(mean_abundance[family])
            residual = len(time_points) - len(mean_abundance[family])
        else:
            columns_t.append(family)
            data_t_tmp.append(mean_abundance[family])
    for i in range(len(data_t_tmp)):
        data_t.append(data_t_tmp[i][residual:])
    time = np.arange(np.array(data_t).shape[1])
    df_t = pd.DataFrame(np.array(data_t).T, columns=columns_t)
    df_o = pd.DataFrame(np.array(data_o).T, columns=columns_t)
    data_dict_t = df_t.to_dict(orient='list')
    categories = list(data_dict_t.keys())
    if types == "Phylum":
        stream_plot(data_dict_t, categories, time, "True", subject, "Phylum")
    else:
        stream_plot(data_dict_t, categories, time, "True", subject, "Genus")
    data_dict_o = df_o.to_dict(orient='list')
    categories = list(data_dict_o.keys())
    if types == "Phylum":
        stream_plot(data_dict_o, categories, time, "Predicted", subject, "Phylum")
    else:
        stream_plot(data_dict_o, categories, time, "Predicted", subject, "Genus")

    iters = 0
    fn = len(families)/2
    high = int(fn*2)
    fig, axes = plt.subplots(nrows=int(fn),ncols=1,sharex=True,figsize=(8,high)) 
    fig.suptitle('Abundance of {types} Level Over Time ({subject})'.format(types=types,subject=subject))
    boxdata = []
    for family in families:
        if "_predict" in family:
            for value in mean_abundance[family]:
                boxdata.append({'Abundance': value, '{types}'.format(types=types): family.split("_predict")[0],'Dataset': "Predicted"})
        else:
            for value in mean_abundance[family]:
                boxdata.append({'Abundance': value, '{types}'.format(types=types): family,'Dataset': "Ground Truth"})
        if iters % 2 == 0:
            print("iters: {0}/{1}".format(int(fn),int(iters/2 +1)))
            means = np.array(mean_abundance[family])
            sems = np.nan_to_num(np.array(sem_abundance[family]), nan=0.0, posinf=0.0, neginf=0.0)
            y1_smooth = means
            y2_smooth = sems
            x_smooth = time_points[len(time_points)-len(y1_smooth):]
            y2_smooth = sems
            axes[int(iters/2)].plot(x_smooth, y1_smooth, label=family, linewidth=3.0, alpha=0.7)
            if types == "Phylum":
                axes[int(iters/2)].fill_between(x_smooth, y1_smooth - 1 * y2_smooth, 
                                y1_smooth + 1 * y2_smooth, alpha=0.1)
        else:
            means = np.array(mean_abundance[family])
            sems = np.nan_to_num(np.array(sem_abundance[family]), nan=0.0, posinf=0.0, neginf=0.0)
            y1_smooth = means
            y2_smooth = sems
            x_smooth = time_points[len(time_points)-len(y1_smooth):]
            y2_smooth = sems
            axes[int(iters/2)].plot(x_smooth, y1_smooth, label=family, linewidth=3.0, alpha=0.7)
            if types == "Phylum":
                axes[int(iters/2)].fill_between(x_smooth, y1_smooth - 0.7 * y2_smooth, 
                                y1_smooth + 0.7 * y2_smooth, alpha=0.4)
            axes[int(iters/2)].legend()
            if int(fn) == int(iters/2+1):
                axes[int(iters/2)].spines['right'].set_visible(False)
                axes[int(iters/2)].spines['top'].set_visible(False)
                axes[int(iters/2)].tick_params(axis='x', labelbottom=True)
                axes[int(iters/2)].set_xlabel('Date (days)')
                base_tick = int(int(len(y1_smooth)/5)/4)
                ticks_to_display = [base_tick*5, base_tick*10, base_tick*15, base_tick*20]
                axes[int(iters/2)].set_xticks(ticks_to_display)
            elif ((int(fn) // 2) + 1) == int(iters/2+1):
                axes[int(iters/2)].set_ylabel('Abundance')
                axes[int(iters/2)].set_xticks([])
                axes[int(iters/2)].spines['right'].set_visible(False)
                axes[int(iters/2)].spines['top'].set_visible(False)
                axes[int(iters/2)].spines['bottom'].set_visible(False)
            else:
                axes[int(iters/2)].tick_params(axis='x', labelbottom=False)
                axes[int(iters/2)].set_xticks([])
                axes[int(iters/2)].spines['right'].set_visible(False)
                axes[int(iters/2)].spines['top'].set_visible(False)
                axes[int(iters/2)].spines['bottom'].set_visible(False)
        iters += 1

    plt.savefig("results/{subject}{lstm_type}-{name}_average_relative_abundance.jpg".format(name=name, subject=subject, lstm_type=lstm_type), dpi=300)
    plt.clf()
    plt.close()

    df = pd.DataFrame(boxdata)
    print(df)
    plt.figure(figsize=(9, 7))
    palette = ['#7B68EE', '#FF4500']
    fig = sns.boxplot(x = '{types}'.format(types=types), y = 'Abundance', data=df, hue='Dataset', hue_order=['Ground Truth', 'Predicted'], palette=palette)
    boxplot = fig.get_figure()
    fig.set_title("Distribution of Ground Truth versus predicted data at the {types} level ({subject}{lstm_type})".format(types=types, lstm_type=lstm_type, subject=subject))
    boxplot.savefig("results/{subject}{lstm_type}-{name}_boxplot.jpg".format(name=name, lstm_type=lstm_type, subject=subject), dpi=300)
    plt.clf()
    plt.close()
    return mean_abundance

class Motion(Dataset):
    def __init__(self, arrays):
        self.arrays = arrays.numpy()

    def __calc__bk(self):
        for i in range(self.arrays.shape[0]):
            for j in range(self.arrays.shape[1]):
                sub_array = self.arrays[i, j, :, 0]
                self.arrays[i, j, :, 0] = sub_array
        return self.arrays
    
    def __calc__(self, target):
        self.target = target.numpy()
        self.target_copy = np.zeros(shape=self.target.shape)
        for i in range(self.arrays.shape[0]):
            for j in range(self.arrays.shape[1]):
                sub_array = self.arrays[i, j, :, 0]
                diff_array = np.diff(sub_array, prepend=sub_array[0])
                for t in range(0, int(self.target.shape[2])):
                    if t == 0:
                        self.target_copy[i,j,t] = self.target[i,j,t] - self.arrays[i, j, -1, 0]
                    else:
                        self.target_copy[i,j,t] = self.target[i,j,t] - self.target[i,j,t-1]
                self.arrays[i, j, :, 0] = diff_array
        return torch.from_numpy(self.arrays), torch.from_numpy(self.target_copy)

def calculate_fluctuation(out_unnormalized, target_unnormalized, input_unnormalized, train_out_unnormalized, training_target_unnormalized, training_input_unnormalized, lstm_type, num_timesteps_output, input, subject):
    train_raw = []
    print(training_input_unnormalized.shape)
    for i in range(0, training_input_unnormalized.shape[0]-1):
        train_raw.append(training_input_unnormalized[i, :, 0, :])
    trains = np.concatenate((np.squeeze(np.array(train_raw)),np.squeeze(np.array(training_input_unnormalized[training_input_unnormalized.shape[0]-1, :, :, :])).transpose((1,0)),training_target_unnormalized[training_target_unnormalized.shape[0]-1, :, :].transpose(1,0)),axis=0)

    test_raw = []
    print(input_unnormalized.shape)
    for i in range(0, input_unnormalized.shape[0]-1):
        test_raw.append(input_unnormalized[i, :, 0, :])
    tests = np.concatenate((np.squeeze(np.array(test_raw)), np.squeeze(np.array(input_unnormalized[input_unnormalized.shape[0]-1, :, :, :])).transpose((1,0))), axis=0)

    test_target = []
    for i in range(0, target_unnormalized.shape[0]):
        test_target.append(target_unnormalized[i, :, int(num_timesteps_output-1)])
    test_target = np.array(test_target)

    print(out_unnormalized.shape)
    test_out = []
    for i in range(0, out_unnormalized.shape[0]):
        test_out.append(out_unnormalized[i, :, int(num_timesteps_output-1)])
    test_out = np.array(test_out)

    train_target = []
    for i in range(0, training_target_unnormalized.shape[0]):
        train_target.append(training_target_unnormalized[i, :, int(num_timesteps_output-1)])
    train_target = np.array(train_target)

    train_out = []
    for i in range(0, train_out_unnormalized.shape[0]):
        train_out.append(train_out_unnormalized[i, :, int(num_timesteps_output-1)])
    train_out = np.array(train_out)

    targets = np.concatenate((trains,test_target),axis=0)
    outs = np.concatenate((trains,test_out),axis=0)
    print(targets.shape)
    print(outs.shape)

    otus = {}
    otu_list = []
    genus = {}
    iters = 0
    with open(input,"r")as handle:
        for i in handle.readlines():
            if iters == 0:
                titles = i.replace("\n","").split(",")[1:]
                iters += 1
            otu_list.append(i.split(",")[0])
    otu_list = otu_list[1:]
    for i in range(0, len(otu_list)):
        if "p__" in otu_list[i]:
            try:
                otus[otu_list[i].split("p__")[1].split(";")[0]].append(i)
            except:
                otus[otu_list[i].split("p__")[1].split(";")[0]] = [i]
        if "g__" in otu_list[i]:
            if otu_list[i].split("p__")[1].split(";")[0] != "":
                try:
                    genus[otu_list[i].split("g__")[1].split(";")[0]].append(i)
                except:
                    genus[otu_list[i].split("g__")[1].split(";")[0]] = [i]

    time_points = []
    for i in range(0,test_target.shape[0]):
        time_points.append(i)
    title = []
    for i in titles:
        try:
            title.append("_".join((i.split("_")[-3],i.split("_")[-2],i.split("_")[-1])))
        except:
            title.append(i)
    title = title[len(title)-test_target.shape[0]:]

    target_data = {}
    out_data = {}
    families = []
    families_all = []
    stream_t = []
    stream_o = []
    columns = []
    for key in otus:
        if key == "":
            continue
        sub_families = []
        sub_data = {}
        sub_target = {}
        sub_out = {}
        families.append(key)
        families_all.append(key)
        families_all.append(key+"_predict")
        sub_families.append(key)
        sub_families.append(key+"_predict")
        t_data = []
        o_data = []
        for i in otus[key]:
            t_data.append((test_target[:, i]))
            a = test_target[:, i]
            b = test_out[:, i]
            if len(time_points) <= 4:
                linear_fit(a=train_out[:, i], b=train_target[:, i], types="train", num_timesteps_output=num_timesteps_output)
                o_data.append((linear_fit(a = test_out[:, i], b=test_target[:, i], types="test", num_timesteps_output=num_timesteps_output)))
            else:
                o_data.append(test_out[:, i])
        
        columns.append(key)
        out_stream = np.mean(np.array(o_data),axis=0)
        target_stream_tmp = np.mean(np.array(t_data),axis=0)
        target_stream = target_stream_tmp[len(target_stream_tmp)-len(out_stream):]
        stream_t.append(list(target_stream))
        stream_o.append(list(out_stream))

        out_data[key+"_predict"] = np.array(o_data)
        target_data[key] = np.array(t_data)
        sub_out[key+"_predict"] = np.array(o_data)
        sub_target[key] = np.array(t_data)
        sub_data = dict(sub_target, **sub_out)
    p_data = dict(target_data, **out_data)
    p_families = ['Proteobacteria', 'Proteobacteria_predict', 'Firmicutes', 'Firmicutes_predict', 'Bacteroidetes', 'Bacteroidetes_predict', 'Actinobacteria', 'Actinobacteria_predict']
    averaged_1(p_data,p_families,time_points,title,"paper_predict_plot", "Phylum", subject, lstm_type)

    target_data = {}
    out_data = {}
    families = []
    families_all = []
    similar = []
    stream_t = []
    stream_o = []
    columns = []
    for key in genus:
        if key == "":
            continue
        sub_families = []
        sub_data = {}
        sub_target = {}
        sub_out = {}
        families.append(key)
        families_all.append(key)
        families_all.append(key+"_predict")
        sub_families.append(key)
        sub_families.append(key+"_predict")
        t_data = []
        o_data = []
        mse = []
        for i in genus[key]:
            t_data.append((test_target[:, i]))
            mse.append(-np.var(test_out[:, i]))
            a = test_target[:, i]
            b = test_out[:, i]

            if len(time_points) <= 4:
                linear_fit(a=train_out[:, i], b=train_target[:, i], types="train", num_timesteps_output=num_timesteps_output)
                o_data.append((linear_fit(a = test_out[:, i], b=test_target[:, i], types="test", num_timesteps_output=num_timesteps_output)))
            else:
                o_data.append((test_out[:, i]))

        columns.append(key)
        out_stream = np.mean(np.array(o_data),axis=0)
        target_stream_tmp = np.mean(np.array(t_data),axis=0)
        target_stream = target_stream_tmp[len(target_stream_tmp)-len(out_stream):]
        stream_t.append(list(target_stream))
        stream_o.append(list(out_stream))

        similar.append(np.sum(mse))
        out_data[key+"_predict"] = np.array(o_data)
        target_data[key] = np.array(t_data)
        sub_out[key+"_predict"] = np.array(o_data)
        sub_target[key] = np.array(t_data)
        sub_data = dict(sub_target, **sub_out)
    sorted_indices = np.argsort(np.array(similar))
    top_indices = sorted_indices[:6]
    g_families_selected = []
    for i in top_indices:
        g_families_selected.append(families_all[i*2])
        g_families_selected.append(families_all[i*2+1])
    g_data = dict(target_data, **out_data)

    averaged_1(g_data,g_families_selected,time_points,title,"paper_genus_predict_plot_full", "Genus", subject, lstm_type)

    return (g_data, p_data, g_families_selected, p_families, time_points, title)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="author:\t{0}\nmail:\t{1}\ndate:\t{2}\n".format(__author__,__mail__,__date__))
    parser.add_argument('--enable-cuda', action='store_true',
                        help='Enable CUDA')
    parser.add_argument('--num_timesteps_input', dest="num_timesteps_input", type=int, default=12, help='[ Optional Default(12) ]: Set the length of the time step used for training.')
    parser.add_argument('--num_timesteps_output', dest='num_timesteps_output', type=int, default=1, help='[ Optional Default(1) ]: Set the length of the time step for the prediction output.')
    parser.add_argument('-i', '--input', dest='input', type=str, help='[ Required ]: Setting the input file.')
    parser.add_argument('-s', '--subject', dest='subject', type=str, help='[ Required ]: Setting the subject name.')
    args = parser.parse_args()
    args.device = None
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    torch.manual_seed(7)

    A = np.load("data/{subject}-adj_mat.npy".format(subject=args.subject))
    with open("data/{subject}-train.pk".format(subject=args.subject),"rb")as handle:
        data = pk.load(handle)
    training_input, training_target, val_input, val_target, means, stds = data[0]+1, data[1]+1, data[2]+1, data[3]+1, data[4], data[5]

    training_input_Motion, training_target_Motion = Motion(training_input).__calc__(training_target)
    val_input_Motion, val_target_Motion = Motion(val_input).__calc__(val_target)

    A_wave = A
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    net = LSTM(A_wave.shape[0],
                training_input.shape[3],
                args.num_timesteps_input,
                args.num_timesteps_output).to(device=args.device)
    lstm_type = "-lstm"
    try:
        net.load_state_dict(torch.load("checkpoints/{subject}{lstm_type}-best_loss.pth".format(subject=args.subject, lstm_type=lstm_type)))
    except:
        net.load_state_dict(torch.load("checkpoints/{subject}{lstm_type}-best_loss.pth".format(subject=args.subject, lstm_type=lstm_type),map_location="cpu"))
    net.eval()

    with torch.no_grad():
        training_input = training_input.to(device=args.device)
        training_input_Motion = training_input_Motion.to(device=args.device)
        training_target = training_target.to(device=args.device)

        val_input = val_input.to(device=args.device)
        val_input_Motion = val_input_Motion.to(device=args.device)
        val_target = val_target.to(device=args.device)

        out = net(A_wave, val_input, val_input_Motion)
        out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
        target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]
        input_unnormalized = val_input.detach().cpu().numpy()*stds[0]+means[0]


        t_out = net(A_wave, training_input, training_input_Motion)
        train_out_unnormalized = t_out.detach().cpu().numpy()*stds[0]+means[0]
        training_target_unnormalized = training_target.detach().cpu().numpy()*stds[0]+means[0]
        training_input_unnormalized = training_input.detach().cpu().numpy()*stds[0]+means[0]
    
    l_g_data, l_p_data, l_g_families_selected, l_p_families, time_points, title = calculate_fluctuation(out_unnormalized, target_unnormalized, input_unnormalized, train_out_unnormalized, training_target_unnormalized, training_input_unnormalized, lstm_type, args.num_timesteps_output, args.input, args.subject)

    net = TwoStreamSTGCN(A_wave.shape[0],
                training_input.shape[3],
                args.num_timesteps_input,
                args.num_timesteps_output).to(device=args.device)
    
    net.load_state_dict(torch.load("checkpoints/{subject}-best_loss.pth".format(subject=args.subject)))
    net.eval()

    with torch.no_grad():
        training_input = training_input.to(device=args.device)
        training_input_Motion = training_input_Motion.to(device=args.device)
        training_target = training_target.to(device=args.device)

        val_input = val_input.to(device=args.device)
        val_input_Motion = val_input_Motion.to(device=args.device)
        val_target = val_target.to(device=args.device)

        out = net(A_wave, val_input, val_input_Motion)
        out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
        target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]
        input_unnormalized = val_input.detach().cpu().numpy()*stds[0]+means[0]


        t_out = net(A_wave, training_input, training_input_Motion)
        train_out_unnormalized = t_out.detach().cpu().numpy()*stds[0]+means[0]
        training_target_unnormalized = training_target.detach().cpu().numpy()*stds[0]+means[0]
        training_input_unnormalized = training_input.detach().cpu().numpy()*stds[0]+means[0]

    g_data, p_data, g_families_selected, p_families, time_points, title = calculate_fluctuation(out_unnormalized, target_unnormalized, input_unnormalized, train_out_unnormalized, training_target_unnormalized, training_input_unnormalized, "", args.num_timesteps_output, args.input, args.subject)

    families_rs = {
        "subject A Saliva" : ["Rothia", "Rothia_predict", "Porphyromonas", "Porphyromonas_predict"],
        "subject A Stool" : ["Bacteroides", "Bacteroides_predict", "Gemmiger", "Gemmiger_predict"],
        "subject M3 Stool" : ["Bacteroides", "Bacteroides_predict", "Faecalibacterium", "Faecalibacterium_predict"],
        "subject M3 Saliva" : ["Veillonella", "Veillonella_predict", "Fusobacterium", "Fusobacterium_predict"],
        "subject B Stool" : ["Bacteroides", "Bacteroides_predict", "Gemmiger", "Gemmiger_predict"],
        "subject F4 Stool" : ["Bacteroides", "Bacteroides_predict", "Akkermansia", "Akkermansia_predict"],
        "subject F4 Saliva" : ["Rothia", "Rothia_predict", "Neisseria", "Neisseria_predict"]
    }
    print(g_families_selected)
    averaged(g_data,l_g_data,families_rs[args.subject], l_g_families_selected,time_points,title,"cat_paper_genus_predict_plot_full", "Genus", args.subject)
    averaged(p_data,l_p_data,p_families, l_p_families,time_points,title,"cat_paper_predict_plot", "Phylum", args.subject)