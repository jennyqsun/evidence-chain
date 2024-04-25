import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, alpha=1, beta=0):
    return 1 / (1 + np.exp(-(alpha*x - beta)))


def get_abs_evidence(x):
    return np.abs(x.cumsum(axis=1))

def get_nback_evidence(x, step=0):
    len_x = x.shape[1]
    ind= len_x - step - 1
    if step >= len_x:
        ind = 0
    x = x[:,ind:]
    return np.abs(x.cumsum(axis=1))

def get_counter(x):
    counter_positive = np.sum(x == 1, axis=1)
    counter_negative = np.sum(x == -1, axis=1)
    paired_counter = [(pos, neg) for pos, neg in zip(counter_positive, counter_negative)]
    return paired_counter

def plot_roc_per_sample_position(position_of_samples, auc_list_train=None, auc_list_test=None, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,1)
    if auc_list_train is not None:
        ax.plot(position_of_samples, auc_list_train,'-o', label = 'training data')
    if auc_list_test is not None:
        ax.plot(position_of_samples, auc_list_test,'-o', label = 'test data')

    ax.set_xlabel("Position of sample")
    ax.set_ylabel("AUC")
    ax.set_ylim(0,1)
    fig.legend()
    return fig, ax

def get_chain_matrix(sequence, count):
    chain_matrix = np.zeros_like(sequence)
    chain_matrix.fill(np.nan)
    for c in range(sequence.shape[0]):
        chain_matrix[c, : int(count[c])] = sequence[c, : int(count[c])]
    return chain_matrix

def make_dataset(num_of_samples, count, chain_matrix):

    dataset = chain_matrix[count>=num_of_samples, :]
    dataset_resp = np.isnan(dataset[:, num_of_samples])  # True means they stopped, false means they kept going
    dataset = dataset[:, :num_of_samples]

    return dataset, dataset_resp


def plot_roc(fpr_list, tpr_list, auc_list, position_of_samples, total_n_list, auc_cutoff=0.5):
    fig, ax = plt.subplots(1,1)
    for i in range(len(fpr_list)):
        if auc_list[i]>=auc_cutoff:
            if i == np.argmax(auc_list):
                ax.plot(fpr_list[i],tpr_list[i],label=f'number of samples: {str(position_of_samples[i])}, auc={np.round(auc_list[i],2)}, n = {total_n_list[i]}', linewidth = 4)
            else:
                ax.plot(fpr_list[i],tpr_list[i],label=f'number of samples: {str(position_of_samples[i])}, auc={np.round(auc_list[i],2)}, n = {total_n_list[i]}')
    ax.plot(np.linspace(0, 1), np.linspace(0, 1), "--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1,1,0,0))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    return fig,ax