import time 
import os 
import gc
import numpy as np
from utils import *
from shapeletnet import *
from sklearn.ensemble import RandomForestClassifier

from rocket.rocket_functions import generate_kernels 
from rocket.rocket_functions import apply_kernels as rocket_applier

dataset_folder = '/home/etud/mbouopda/Univariate_arff' # the folder containing the datasets

nb_run_per_dataset = 5 # number of run for each dataset

columns = ('classifier','dataset', 'acc_mean', 'acc_std', 'train_time_in_sec_mean', 'train_time_in_sec_std', 'test_time_in_sec_mean', 'test_time_in_sec_std') # columns of the output file

dataset_names_file = "dataset_names.txt" # file containing the names of the datasets to run experiments on

min_shp_length = 3 # min candidate length

nb_inst_per_class = 1 # the number of instances to use per class in order to generate shapelet candidate

datasets = np.loadtxt(dataset_names_file, dtype=str) # every dataset names

to_skip = np.loadtxt('skip.txt', dtype=str) # model-data to skip 

output_file = 'results.csv' # the name of the result file

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

			accuracy_1S2 = []
			train_time_1S2 = []
			test_time_1S2 = []

			accuracy_rocket = []
			train_time_rocket = []
			test_time_rocket = []

			# free some memory
			del train_ds
			del test_ds
			
			print('Executing:', dataset)
			for _ in range(nb_run_per_dataset):

				gc.collect()
				# ----------- 1S2----------------------
				if f'1S2{dataset}' not in to_skip:
			
					kernelsGenerator = KernelsGenerator(cand_length_list=np.arange(min_shp_length, max_shp_length+1), 
					                          nb_inst_per_class=nb_inst_per_class, 
					                          random_state=None)
					kernelsGenerator.generate(X_train, y_train)

					train_start = time.time()

					clf = RandomForestClassifier()

					X_train_transformed = apply_kernels(X_train, kernelsGenerator.kernels)

					clf.fit(X_train_transformed, y_train)

					train_time_1S2.append(time.time() - train_start)

					test_start = time.time()
					
					X_test_transformed = apply_kernels(X_test, kernelsGenerator.kernels)
					acc = clf.score(X_test_transformed, y_test)

					test_time_1S2.append(time.time() - test_start)

					accuracy_1S2.append(acc)


				# ----------------- Rocket -----------------------
				if f'Rocket{dataset}' not in to_skip:
					kernels = generate_kernels(X_train.shape[-1], 10_000)

					train_start = time.time()

					clf = RandomForestClassifier()

					X_train_transformed = rocket_applier(X_train, kernels)

					clf.fit(X_train_transformed, y_train)

					train_time_rocket.append(time.time() - train_start)

					test_start = time.time()
					
					X_test_transformed = rocket_applier(X_test, kernels)
					acc = clf.score(X_test_transformed, y_test)

					test_time_rocket.append(time.time() - test_start)

					accuracy_rocket.append(acc)

			if len(accuracy_1S2) > 0:
				result = ['1S2', dataset, np.mean(accuracy_1S2), np.std(accuracy_1S2), np.mean(train_time_1S2), np.std(train_time_1S2), np.mean(test_time_1S2), np.std(test_time_1S2)]
				f.write(','.join(np.array(result, dtype=np.str)) + '\n')

			if len(accuracy_rocket) > 0:
				result = ['Rocket', dataset, np.mean(accuracy_rocket), np.std(accuracy_rocket), np.mean(train_time_rocket), np.std(train_time_rocket), np.mean(test_time_rocket), np.std(test_time_rocket)]
				f.write(','.join(np.array(result, dtype=np.str)) + '\n')
			
			gc.collect()
		except Exception as e:
			print('\n###########\nFailed on dataset:', dataset, ' Msg:', e, '\n##################\n')

print('All done')

		


