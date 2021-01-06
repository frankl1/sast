import time 
import os 
import gc
import numpy as np
from utils import *
from sast import *
from sklearn.linear_model import RidgeClassifierCV

dataset_folder = '/home/etud/mbouopda/Univariate_arff' # the folder containing the datasets

nb_run_per_dataset = 5 # number of run for each dataset

columns = ('classifier','dataset', 'acc_mean', 'acc_std', 'train_time_in_sec_mean', 'train_time_in_sec_std', 'test_time_in_sec_mean', 'test_time_in_sec_std') # columns of the output file

dataset_names_file = "dataset_names_small.txt" # file containing the names of the datasets to run experiments on

min_shp_length = 3 # min candidate length

nb_inst_per_class = 1 # the number of instances to use per class in order to generate shapelet candidate

datasets = np.loadtxt(dataset_names_file, dtype=str) # every dataset names

to_skip = np.loadtxt('skip-sast-ensemble.txt', dtype=str) # model-data to skip 

output_file = 'results-sast-ensemble.csv' # the name of the result file

combination_list = [list(range(3, 10)), list(range(10, 17)), list(range(17, 24))]

print('Running on', len(datasets), ' datasets')
print('Number of run per dataset:', nb_run_per_dataset)

mode = 'w+'
if os.path.isfile(output_file):
	mode = 'a+'

with open(output_file, mode, buffering=1) as f:

	if mode == 'w+':
		f.write(','.join(columns) + '\n')

	for i, dataset in enumerate(datasets):

		try:
			train_ds, test_ds = load_dataset(dataset_folder, dataset)
			
			# fill na
			train_ds.fillna(0, axis=1, inplace=True)
			test_ds.fillna(0, axis=1, inplace=True)
			
			X_train, y_train = format_dataset(train_ds, shuffle=True)
			X_test, y_test = format_dataset(test_ds)

			max_shp_length = X_train.shape[1]

			accuracy_saste_ridge = []
			train_time_saste_ridge = []
			test_time_saste_ridge = []

			# free some memory
			del train_ds
			del test_ds

			combi_list = list(filter(lambda cb: min(cb) <= X_train.shape[1], combination_list))
			
			print('Executing:', dataset)
			for _ in range(nb_run_per_dataset):

				gc.collect()
				
				# ----------------- SAST with ridge -----------------------
				if f'SASTE-Ridge{dataset}' not in to_skip:

					clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
					saste = SASTEnsemble(cand_length_list=combi_list,
					                          nb_inst_per_class=nb_inst_per_class, 
					                          random_state=None, classifier=clf, n_jobs = -1)

					train_start = time.time()

					saste.fit(X_train, y_train)

					train_time_saste_ridge.append(time.time() - train_start)

					test_start = time.time()
					
					acc = saste.score(X_test, y_test)

					test_time_saste_ridge.append(time.time() - test_start)

					accuracy_saste_ridge.append(acc)

			if len(accuracy_saste_ridge) > 0:
				result = ['SASTE-Ridge', dataset, np.mean(accuracy_saste_ridge), np.std(accuracy_saste_ridge), np.mean(train_time_saste_ridge), np.std(train_time_saste_ridge), np.mean(test_time_saste_ridge), np.std(test_time_saste_ridge)]
				f.write(','.join(np.array(result, dtype=np.str)) + '\n')
			
			gc.collect()
		except Exception as e:
			print('\n###########\nFailed on dataset:', dataset, ' Msg:', e, '\n##################\n')

print('All done')
