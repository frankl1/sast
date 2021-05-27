import time 
import os 
import gc
import numpy as np
import pandas as pd
import sys 

sys.path.append(os.path.abspath('.'))

from joblib import Parallel, delayed

from sast.utils import *
from sast.sast import *
from sast.stck import ShapeletTransformK

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV

from sktime.utils.data_processing import from_2d_array_to_nested
from sktime.transformations.panel.shapelets import ContractedShapeletTransform

dataset_folder = '/home/local.isima.fr/mimbouop/Univariate_arff' # the folder containing the datasets

columns = ('classifier','dataset', 'acc_mean', 'acc_std', 'train_time_in_sec_mean', 'train_time_in_sec_std', 'test_time_in_sec_mean', 'test_time_in_sec_std') # columns of the output file

nb_run_per_dataset = 5 # number of run for each dataset

dataset_names_file = "dataset_names.txt" # file containing the names of the datasets to run experiments on

min_shp_length = 3 # min candidate length

max_shp_length = np.inf # max candidate length

nb_inst_per_class = (0, 1, 2, 5, 9, 0.25, 0.5, 0.75) # the number of instances to use per class in order to generate shapelet candidate, 0 to average per class

datasets = np.loadtxt(dataset_names_file, dtype=str, ndmin=1) # every dataset names

output_file = 'results/results-stc-k.csv' # the name of the result file

time_contract_in_mins = 60 # the time contract for STC

n_jobs = -1 # number of workers

print('Running on', len(datasets), ' datasets')
print('Number of run per dataset:', nb_run_per_dataset)

def run_stc(X_train, X_test, y_train, y_test):
	accuracy = []
	train_time = []
	test_time = []
	
	for _ in range(nb_run_per_dataset):
		try:
			st = ContractedShapeletTransform(min_shapelet_length=min_shp_length, 
				max_shapelet_length=np.inf, 
			time_contract_in_mins=time_contract_in_mins)
			clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

			start_time = time.time()

			st.fit(X_train, y_train)

			X_train_transformed = st.transform(X_train)

			clf.fit(X_train_transformed, y_train)

			fit_time = time.time()

			X_test_transformed = st.transform(X_test)

			accuracy.append(clf.score(X_test_transformed, y_test))

			end_time = time.time()

			train_time.append(fit_time - start_time)
			test_time.append(end_time - fit_time)
		except Exception as e:
			print('STC Exception on current dataset with message:', e)

	if len(accuracy) > 0:
		return np.mean(accuracy), np.std(accuracy), np.mean(train_time), np.std(train_time), np.mean(test_time), np.std(test_time)
	else:
		return None, None, None, None, None, None

def run_stc_k(X_train, X_test, y_train, y_test, nb_inst_per_class):
	accuracy = []
	train_time = []
	test_time = []

	for _ in range(nb_run_per_dataset):
		try:
			st = ShapeletTransformK(min_shapelet_length=min_shp_length, 
		                          max_shapelet_length=np.inf, 
		                          nb_inst_per_class=nb_inst_per_class,
		                          time_contract_in_mins=time_contract_in_mins,
		                          avg_per_class=nb_inst_per_class == 0
		                         )

			clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

			start_time = time.time()

			st.fit(X_train, y_train)

			X_train_transformed = st.transform(X_train)

			clf.fit(X_train_transformed, y_train)

			fit_time = time.time()

			X_test_transformed = st.transform(X_test)

			accuracy.append(clf.score(X_test_transformed, y_test))

			end_time = time.time()

			train_time.append(fit_time - start_time)
			test_time.append(end_time - fit_time)
		except Exception as e:
			print(f'STC-{nb_inst_per_class} Exception on current dataset with message:', e)

	if len(accuracy) > 0:
		return np.mean(accuracy), np.std(accuracy), np.mean(train_time), np.std(train_time), np.mean(test_time), np.std(test_time)
	else:
		return None, None, None, None, None, None

def run_on_dataset(dataset):
	train_ds, test_ds = load_dataset(dataset_folder, dataset)
			
	# fill na
	train_ds.fillna(0, axis=1, inplace=True)
	test_ds.fillna(0, axis=1, inplace=True)
	
	X_train, y_train = format_dataset(train_ds, shuffle=True)
	X_test, y_test = format_dataset(test_ds)

	X_train_sktime = from_2d_array_to_nested(pd.DataFrame(X_train))
	X_test_sktime = from_2d_array_to_nested(pd.DataFrame(X_test))
	
	try:
		nipc_filtered = list(filter(lambda k: f'STC-{k}{dataset}' not in to_skip, nb_inst_per_class)) 
		if len(nipc_filtered) > 0:
			print('Running STC-k on', dataset, 'for k in', nipc_filtered)
			res = Parallel(n_jobs=n_jobs)(delayed(run_stc_k)(X_train_sktime, X_test_sktime, y_train, y_test, nb_inst_per_class=nbipc) for nbipc in nipc_filtered)
			
			content = ""
			for i, (acc_mean, acc_std, train_mean, train_std, test_mean, test_std) in enumerate(res):
				nbipc = nipc_filtered[i]
				result = [f'STC-{nbipc}', dataset, acc_mean, acc_std, train_mean, train_std, test_mean, test_std]
				content += ','.join(np.array(result, dtype=np.str)) + '\n'
			
			with open(output_file, mode, buffering=1) as f:
				f.write(content)
	except Exception as e:
		print('\n###########\n STC-k Failed on dataset:', dataset, ' Msg:', e, '\n##################\n')


	try:	
		if f'STC{dataset}' in to_skip:
			return

		print('Running STC on ', dataset)
		acc_mean, acc_std, train_mean, train_std, test_mean, test_std = run_stc(X_train_sktime, X_test_sktime, y_train, y_test)

		result = [f'STC', dataset, acc_mean, acc_std, train_mean, train_std, test_mean, test_std]
		
		with open(output_file, mode, buffering=1) as f:
			f.write(','.join(np.array(result, dtype=np.str)) + '\n')

	except Exception as e:
		print('\n###########\n STC Failed on dataset:', dataset, ' Msg:', e, '\n##################\n')


to_skip = []

mode = 'w+'
if os.path.isfile(output_file):
	mode = 'a+'
	to_skip = np.loadtxt(output_file, usecols=(0,1), dtype=str, delimiter=',', skiprows=1)
	to_skip = [''.join(elt) for elt in to_skip]

with open(output_file, mode, buffering=1) as f:

	if mode == 'w+':
		f.write(','.join(columns) + '\n')

	batch_size = 12 

	for i in range(0, len(datasets), batch_size):
		Parallel(n_jobs=n_jobs)(delayed(run_on_dataset)(dataset) for dataset in datasets[i:i+batch_size])

print('All done')
