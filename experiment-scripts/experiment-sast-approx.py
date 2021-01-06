import time 
import os 
import gc
import itertools
import numpy as np
from utils import *
from sast import *
from sklearn.linear_model import RidgeClassifierCV

dataset_folder = '/home/etud/mbouopda/Univariate_arff' # the folder containing the datasets

columns = ('classifier','dataset', 'acc_mean', 'acc_std', 'train_time_in_sec_mean', 'train_time_in_sec_std', 'test_time_in_sec_mean', 'test_time_in_sec_std') # columns of the output file

nb_run_per_dataset = 5 # number of run for each dataset

dataset_names_file = "dataset_names_small.txt" # file containing the names of the datasets to run experiments on

nb_inst_per_class = 1 # the number of instances to use per class in order to generate shapelet candidate

datasets = np.loadtxt(dataset_names_file, dtype=str) # every dataset names

to_skip = np.loadtxt('skip-sast-fixed.txt', dtype=str) # cand length - dataset to skip 

output_file = 'results-sast-fixed.csv' # the name of the result file

print('Running on', len(datasets), ' datasets')
print('Number of run per dataset:', nb_run_per_dataset)

mode = 'w+'
if os.path.isfile(output_file):
	mode = 'a+'

# list of length_list
combinations = [(9, 13, 15), (7, 11, 15), (7, 9, 15), (9, 11, 15)]

with open(output_file, mode, buffering=1) as f:

	if mode == 'w+':
		f.write(','.join(columns) + '\n')

	for length_list in combinations:

		for i, dataset in enumerate(datasets):
			tmp_ridge = '_'.join(np.array(length_list, dtype=str))+'-SAST-Ridge' + dataset

			try:
				train_ds, test_ds = load_dataset(dataset_folder, dataset)
				
				# fill na
				train_ds.fillna(0, axis=1, inplace=True)
				test_ds.fillna(0, axis=1, inplace=True)
				
				X_train, y_train = format_dataset(train_ds, shuffle=True)
				X_test, y_test = format_dataset(test_ds)

				# free some memory
				del train_ds
				del test_ds
				gc.collect()

				print('Executing:', dataset)

				accuracy_ridge = []
				train_time_ridge = []
				test_time_ridge = []

				for _ in range(nb_run_per_dataset):

					## using ridge
					if tmp_ridge not in to_skip:
						clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
						sast_ridge = SAST(cand_length_list=length_list,
			                          nb_inst_per_class=nb_inst_per_class, 
			                          random_state=None, classifier=clf)

						train_start = time.time()

						sast_ridge.fit(X_train, y_train)

						train_time_ridge.append(time.time() - train_start)

						test_start = time.time()
						
						acc = sast_ridge.score(X_test, y_test)

						test_time_ridge.append(time.time() - test_start)

						accuracy_ridge.append(acc)

				result = ['_'.join(np.array(length_list, dtype=str))+'-SAST-Ridge', dataset, np.mean(accuracy_ridge), np.std(accuracy_ridge), np.mean(train_time_ridge), np.std(train_time_ridge), np.mean(test_time_ridge), np.std(test_time_ridge)]
				f.write(','.join(np.array(result, dtype=np.str)) + '\n')

			except Exception as e:
				print('\n###########\nFailed on dataset:', dataset, ' Msg:', e, '\n##################\n')

print('All done')
