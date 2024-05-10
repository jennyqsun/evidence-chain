import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from color_util import *

##### equations ################



def sigmoid(x, alpha=1, beta=0):
    return 1 / (1 + np.exp(-(alpha * x - beta)))


def get_gmeans(tpr, fpr):
    return np.sqrt(tpr * (1 - fpr))


##### process chain ############
def get_abs_evidence(x):
        return np.abs(x.cumsum(axis=1))

def get_evidence(x):
        x_cumsum = x.cumsum(axis=1)
        index_positive, index_negative = x_cumsum[:,-1]>=0, x_cumsum[:,-1]<0
        x_cumsum[index_negative] = x_cumsum[index_negative] * -1
        return x_cumsum 


def get_race_counters(x_train):
    counter_1 = np.sum(x_train == 1, axis=1)
    counter_2 = np.sum(x_train != 1, axis=1)
    return counter_1, counter_2


def get_nback_evidence(x, step=0):
    len_x = x.shape[1]
    ind = len_x - step - 1
    if step >= len_x:
        ind = 0
    x = x[:, ind:]
    return np.abs(x.cumsum(axis=1))


def get_counter(x):
    counter_positive = np.sum(x == 1, axis=1)
    counter_negative = np.sum(x == -1, axis=1)
    paired_counter = [
        (pos, neg) for pos, neg in zip(counter_positive, counter_negative)
    ]
    return paired_counter


def get_max_runs(x_train):
    runs = []
    for chain in x_train:
        result_p, result_n, counter_p, counter_n = 0, 0, 0, 0
        for n in chain:
            counter_p = (counter_p + 1) if n == 1 else 0
            result_p = max(counter_p, result_p)
            counter_n = (counter_n + 1) if n == -1 else 0
            result_n = max(counter_n, result_n)
        max_run = max(result_p, result_n)
        runs.append(max_run)
    return runs


def get_chain_matrix(sequence, count):
    chain_matrix = np.zeros_like(sequence)
    chain_matrix.fill(np.nan)
    for c in range(sequence.shape[0]):
        chain_matrix[c, : int(count[c]) + 1] = sequence[c, : int(count[c]) + 1]
    return chain_matrix


def make_dataset(stop_sample, count, chain_matrix):
    # stop sample would go to 28 at most, because that means subject see 29 samples and stopped at 30th
    dataset = chain_matrix[count >= stop_sample, :]
    dataset_resp = np.isnan(
        dataset[:, stop_sample + 1]
    )  # True means they stopped, false means they kept going
    dataset = dataset[:, : stop_sample + 1]

    return dataset, dataset_resp


def correct_samples_by_condition(df):
    df["count_corrected"] = df["count"]
    df.loc[df["stimDur"] == 0.25, "count_corrected"] = (
        df.loc[df["stimDur"] == 0.25, "count"] - 1
    )
    df.loc[df["stimDur"] == 0.1, "count_corrected"] = (
        df.loc[df["stimDur"] == 0.1, "count"] - 2
    )
    df.loc[df["stimDur"] == 0.05, "count_corrected"] = (
        df.loc[df["stimDur"] == 0.05, "count"] - 3
    )
    df.loc[df["count"] == 30, "count_corrected"] = 29
    # this is saying that for subjects who responded after the 30th sample disappear, they are coded as count ==30
    # because that is the 31st position. 0 is the first sample.
    return df


###### plotting functions  ############
def plot_roc(
    fpr_list,
    tpr_list,
    auc_list,
    position_of_samples,
    total_n_list,
    auc_cutoff=0.5,
    label="number of samples", ax=None, fig=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax =ax
    color_list = get_color_list()

    for i in range(len(fpr_list)):
        if auc_list[i] >= auc_cutoff:
            if i == np.argmax(auc_list):
                ax.plot(
                    fpr_list[i],
                    tpr_list[i],
                    label=f"{label}: {str(position_of_samples[i])}, auc={np.round(auc_list[i],2)}, n = {total_n_list[i]}",
                    linewidth=4,
                    color=color_list[i]
                )
            else:
                ax.plot(
                    fpr_list[i],
                    tpr_list[i],
                    label=f"{label}: {str(position_of_samples[i])}, auc={np.round(auc_list[i],2)}, n = {total_n_list[i]}",
                    color=color_list[i]

                )
    ax.plot(np.linspace(0, 1), np.linspace(0, 1), "--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1, 1, 0, 0))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    return fig, ax


def plot_roc_per_sample_position(
    position_of_samples, auc_list_train=None, auc_list_test=None, fig=None, ax=None
):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)
    if auc_list_train is not None:
        ax.plot(
            [i + 1 for i in position_of_samples], auc_list_train, "-o", label="Trainig"
        )
    if auc_list_test is not None:
        ax.plot([i + 1 for i in position_of_samples], auc_list_test, "-o", label="Test")

    # ax.set_xticks([i+1 for i in position_of_samples])

    ax.set_xlabel("Sample")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    fig.legend()
    return fig, ax
