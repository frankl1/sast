import time 
import os 
import gc
import numpy as np
from utils import *
from shapeletnet import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV

dataset_folder = '/home/etud/mbouopda/Univariate_arff' # the folder containing the datasets

nb_run_per_dataset = 5 # number of run for each dataset

columns = ('classifier','dataset', 'acc_mean', 'acc_std', 'train_time_in_sec_mean', 'train_time_in_sec_std', 'test_time_in_sec_mean', 'test_time_in_sec_std') # columns of the output file

dataset_names_file = "dataset_names_small.txt" # file containing the names of the datasets to run experiments on

min_shp_length = 3 # min candidate length

nb_inst_per_class = 1 # the number of instances to use per class in order to generate shapelet candidate

datasets = np.loadtxt(dataset_names_file, dtype=str) # every dataset names

to_skip = np.loadtxt('skip-rf-ridge.txt', dtype=str) # model-data to skip 

output_file = 'results-rf-ridge.csv' # the name of the result file

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

			accuracy_sast_rf = []
			train_time_sast_rf = []
			test_time_sast_rf = []

			accuracy_sast_ridge = []
			train_time_sast_ridge = []
			test_time_sast_ridge = []

			# free some memory
			del train_ds
			del test_ds
			
			print('Executing:', dataset)
			for _ in range(nb_run_per_dataset):

				gc.collect()
				kernelsGenerator = KernelsGenerator(cand_length_list=np.arange(min_shp_length, max_shp_length+1), 
					                          nb_inst_per_class=nb_inst_per_class, 
					                          random_state=None)
				kernelsGenerator.generate(X_train, y_train)
				
				# ----------- SAST with random forest----------------------
				if f'SAST-RF{dataset}' not in to_skip:

					train_start = time.time()

					clf = RandomForestClassifier(max_features=None, min_impurity_decrease=0.05)

					X_train_transformed = apply_kernels(X_train, kernelsGenerator.kernels)

					clf.fit(X_train_transformed, y_train)

					train_time_sast_rf.append(time.time() - train_start)

					test_start = time.time()
					
					X_test_transformed = apply_kernels(X_test, kernelsGenerator.kernels)
					acc = clf.score(X_test_transformed, y_test)

					test_time_sast_rf.append(time.time() - test_start)

					accuracy_sast_rf.append(acc)


				# ----------------- SAST with ridge -----------------------
				if f'SAST-Ridge{dataset}' not in to_skip:

					train_start = time.time()

					clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

					X_train_transformed = apply_kernels(X_train, kernelsGenerator.kernels)

					clf.fit(X_train_transformed, y_train)

					train_time_sast_ridge.append(time.time() - train_start)

					test_start = time.time()
					
					X_test_transformed = apply_kernels(X_test, kernelsGenerator.kernels)
					acc = clf.score(X_test_transformed, y_test)

					test_time_sast_ridge.append(time.time() - test_start)

					accuracy_sast_ridge.append(acc)

			if len(accuracy_sast_rf) > 0:
				result = ['SAST-RF', dataset, np.mean(accuracy_sast_rf), np.std(accuracy_sast_rf), np.mean(train_time_sast_rf), np.std(train_time_sast_rf), np.mean(test_time_sast_rf), np.std(test_time_sast_rf)]
				f.write(','.join(np.array(result, dtype=np.str)) + '\n')

			if len(accuracy_sast_ridge) > 0:
				result = ['SAST-Ridge', dataset, np.mean(accuracy_sast_ridge), np.std(accuracy_sast_ridge), np.mean(train_time_sast_ridge), np.std(train_time_sast_ridge), np.mean(test_time_sast_ridge), np.std(test_time_sast_ridge)]
				f.write(','.join(np.array(result, dtype=np.str)) + '\n')
			
			gc.collect()
		except Exception as e:
			print('\n###########\nFailed on dataset:', dataset, ' Msg:', e, '\n##################\n')

print('All done')