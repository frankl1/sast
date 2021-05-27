import time 
import os 
import gc
import numpy as np

import sys
sys.path.append(os.path.abspath("."))

from sast.utils import *
from sast.sast import *

from sklearn.linear_model import RidgeClassifierCV

from sktime.utils.data_processing import from_2d_array_to_nested
from sktime.transformations.panel.shapelets import *

dataset_folder = '/home/local.isima.fr/mimbouop/Univariate_arff' # the folder containing the datasets
dataset = 'HouseTwenty' # the dataset to use

columns = ('series_length','SAST-train', 'SAST-test', 'SAST-appox-train', 'SAST-approx-test', 'SASTE-train', 'SASTE-test', 'SASTE-approx-train', 'SASTE-approx-test', 'STC-train', 'STC-test') # columns of the output file

min_shp_length = 3 # min candidate length

output_file = 'results/results-scalability-series-length-contracted.csv' # the name of the result file

max_exponent = 10 # go up to series of length 2^max_exponent

mode = 'w+'
if os.path.isfile(output_file):
	mode = 'a+'

with open(output_file, mode, buffering=1) as f:

	if mode == 'w+':
		f.write(','.join(columns) + '\n')

	train_ds, test_ds = load_dataset(dataset_folder, dataset)
			
	# fill na
	train_ds.fillna(0, axis=1, inplace=True)
	test_ds.fillna(0, axis=1, inplace=True)
	
	X_train_all, y_train = format_dataset(train_ds, shuffle=True)
	X_test_all, y_test_all = format_dataset(test_ds)

	for i in range(4, max_exponent+1):

		series_length = 2**i

		X_train = X_train_all[:,:series_length]
		X_test = X_test_all[:X_train.shape[0], :series_length]
		y_test = y_test_all[:X_train.shape[0]]

		max_shp_length = X_train.shape[1]

		## SAST

		clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
		sast = SAST(cand_length_list=np.arange(min_shp_length, max_shp_length+1),
	                          nb_inst_per_class=1, 
	                          random_state=None, classifier=clf)

		start = time.time()

		sast.fit(X_train, y_train)

		sast_time_fit = time.time() - start

		sast.score(X_test, y_test)

		sast_time_test = time.time() - sast_time_fit

		## SAST Fixed (3)
		clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
		sast_fixed = SAST(cand_length_list=[7, 11, 15],
	                          nb_inst_per_class=1, 
	                          random_state=None, classifier=clf)

		start = time.time()

		sast_fixed.fit(X_train, y_train)

		sast_fixed_time_fit = time.time() - start

		sast_fixed.score(X_test, y_test)

		sast_fixed_time_test = time.time() - sast_fixed_time_fit

		## SASTEnsemble (3)
		combination_list = [np.arange(min_shp_length, max_shp_length+1) for i in range(3)]

		clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
		saste = SASTEnsemble(cand_length_list=combination_list,
	                          nb_inst_per_class=1, 
	                          random_state=None, classifier=clf, n_jobs = -1)

		start = time.time()

		saste.fit(X_train, y_train)

		saste_time_fit = time.time() - start

		saste.score(X_test, y_test)

		saste_time_test = time.time() - saste_time_fit

		## SASTEnsemble Fixed (3)
		combination_list = [list(range(min_shp_length + i, min_shp_length + i + 7)) for i in range(max_shp_length - min_shp_length - 7 + 1)]

		if len(combination_list) > 0:
			clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
			saste_fixed = SASTEnsemble(cand_length_list=combination_list,
		                          nb_inst_per_class=1, 
		                          random_state=None, classifier=clf, n_jobs = -1)

			start = time.time()

			saste_fixed.fit(X_train, y_train)

			saste_fixed_time_fit = time.time() - start

			saste_fixed.score(X_test, y_test)

			saste_fixed_time_test = time.time() - saste_fixed_time_fit
		else:
			saste_fixed_time_fit = np.nan
			saste_fixed_time_test = np.nan

		## ShapeletTransform
		X_train_sktime = from_2d_array_to_nested(pd.DataFrame(X_train))
		X_test_sktime = from_2d_array_to_nested(pd.DataFrame(X_test))
		
		clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

		stc = ContractedShapeletTransform(min_shapelet_length=min_shp_length, max_shapelet_length=np.inf, time_contract_in_mins=60)
		
		start = time.time()
		
		stc.fit(X_train_sktime, y_train)
		X_train_transformed = stc.transform(X_train_sktime)
		clf.fit(X_train_transformed, y_train)

		stc_time_fit = time.time() - start

		X_test_transformed = stc.transform(X_test_sktime)
		clf.score(X_test_transformed, y_test)

		stc_time_test = time.time() - stc_time_fit

		#  Write results
		result = [series_length, sast_time_fit, sast_time_test, sast_fixed_time_fit, sast_fixed_time_test, saste_time_fit, saste_time_test, saste_fixed_time_fit, saste_fixed_time_test, stc_time_fit, stc_time_test]
		f.write(','.join(np.array(result, dtype=np.str)) + '\n')

print('All done')
