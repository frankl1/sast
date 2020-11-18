import os
import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from shapeletnet import znormalize_array 

def load_arff_2_dataframe(fname):
    data = loadarff(fname)
    data = pd.DataFrame(data[0])
    data = data.astype(np.float64)
    return data
    
def load_dataset(ds_folder, ds_name):
    # dataset path
    ds_path = os.path.join(ds_folder, ds_name)
    
    # load train and test set from arff
    train_ds = load_arff_2_dataframe(os.path.join(ds_path, f'{ds_name}_TRAIN.arff'))
    test_ds = load_arff_2_dataframe(os.path.join(ds_path, f'{ds_name}_TEST.arff'))
    
    return train_ds, test_ds

def format_dataset(data, shuffle=True):
    X = data.values.copy()
    if shuffle:
        np.random.shuffle(X)
    X, y = X[:, :-1], X[:, -1]

    return X, y.astype(int)

def plot(h, h_val, title):
    plt.title(title)
    plt.plot(h, label='Train')
    plt.plot(h_val, label='Validation')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def plot_most_important_features(kernels, scores, limit = 4):
    features = zip(kernels, scores)
    sorted_features = sorted(features, key=itemgetter(1), reverse=True)
    for sf in sorted_features[:limit]:
        kernel, score = sf
        kernel = kernel[~np.isnan(kernel)]
        plt.plot(range(kernel.size), kernel, linewidth=50*score, label=f'{score:.5}')
    plt.legend()
    plt.show()
    
def plot_most_important_feature_on_ts(ts, label, features, scores, offset=0, limit = 5, fname=None):
    '''Plot the most important features on ts'''
    features = zip(features, scores)
    sorted_features = sorted(features, key=itemgetter(1), reverse=True)
    
    max_ = min(limit, len(sorted_features) - offset)

    if max_ <= 0:
        print('Nothing to plot')
        return
    fig, axes = plt.subplots(1, max_, sharey=True, figsize=(3*max_, 3), tight_layout=True)
    
    for f in range(max_):
        kernel, score = sorted_features[f+offset]
        kernel_normalized = znormalize_array(kernel)
        d_best = np.inf
        for i in range(ts.size - kernel.size):
            d = np.sum((znormalize_array(ts[i:i+kernel.size]) - kernel_normalized)**2)
            if d < d_best:
                d_best = d
                start_pos = i
        axes[f].plot(range(start_pos, start_pos + kernel.size), kernel, linewidth=5)
        axes[f].plot(range(ts.size), ts, linewidth=2)
        axes[f].set_title(f'feature: {f+1+offset}')
    fig.suptitle(f'Ground truth class: {label}', fontsize=15)
    plt.show();

    if fname is not None:
        fig.savefig(fname)

def plot_kernel_generators(sastClf):
    ''' This herper function is used to plot the reference time series used by a SAST'''
    for c, ts in sastClf.kernels_generators_.items():
        plt.figure(figsize=(5, 3))
        plt.title(f'Class {c}')
        for t in ts:
            plt.plot(t)
        plt.show()