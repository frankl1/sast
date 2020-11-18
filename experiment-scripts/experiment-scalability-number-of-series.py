import time 
import os 
import gc
import numpy as np

from utils import *
from sast import *

from sklearn.linear_model import RidgeClassifierCV

from sktime.utils.data_container import detabularise
from sktime.transformers.series_as_features.shapelets import ShapeletTransform

dataset_folder = '/home/etud/mbouopda/Univariate_arff' # the folder containing the datasets
dataset = 'Chinatown' # the dataset to use

columns = ('number_of_series','SAST', 'SAST-fixed', 'SASTE', 'SASTE-fixed', 'STC') # columns of the output file

min_shp_length = 3 # min candidate length

output_file = 'results-scalability-number-of-series.csv' # the name of the result file

max_exponent = 20 # go up to series of length 2^max_exponent

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
	
	X_train_base, y_train_base = format_dataset(train_ds, shuffle=True)
	X_test, y_test = format_dataset(test_ds)

	for i in range(4, max_exponent+1):

		nb_series = 2**i
		
		if nb_series <= 2048:
			continue

		idx = np.random.choice(np.arange(len(y_train_base)), size=nb_series, replace=True)
		X_train = X_train_base[idx]
		y_train = y_train_base[idx]

		print('Shape:', X_train.shape)

		max_shp_length = X_train.shape[1]

		## SAST
		clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
		sast = SAST(cand_length_list=np.arange(min_shp_length, max_shp_length+1),
	                          nb_inst_per_class=1, 
	                          random_state=None, classifier=clf)

		start = time.time()

		sast.fit(X_train, y_train)
		sast.score(X_test, y_test)

		sast_time = time.time() - start

		## SAST Fixed (3)
		clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
		sast_fixed = SAST(cand_length_list=[7, 11, 15],
	                          nb_inst_per_class=1, 
	                          random_state=None, classifier=clf)

		start = time.time()

		sast_fixed.fit(X_train, y_train)
		sast_fixed.score(X_test, y_test)

		sast_fixed_time = time.time() - start

		## SASTEnsemble (3)
		combination_list = [np.arange(min_shp_length, max_shp_length+1) for i in range(3)]

		clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
		saste = SASTEnsemble(cand_length_list=combination_list,
	                          nb_inst_per_class=1, 
	                          random_state=None, classifier=clf, n_jobs = -1)

		start = time.time()

		saste.fit(X_train, y_train)
		saste.score(X_test, y_test)

		saste_time = time.time() - start

		## SASTEnsemble Fixed (3)
		combination_list = [list(range(3, 10)), list(range(10, 17)), list(range(17, 24))]

		clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
		saste_fixed = SASTEnsemble(cand_length_list=combination_list,
	                          nb_inst_per_class=1, 
	                          random_state=None, classifier=clf, n_jobs = -1)

		start = time.time()

		saste_fixed.fit(X_train, y_train)
		saste_fixed.score(X_test, y_test)

		saste_fixed_time = time.time() - start

		## ShapeletTransform
		X_train_sktime = detabularise(pd.DataFrame(X_train))
		X_test_sktime = detabularise(pd.DataFrame(X_test))
		
		clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

		stc = ShapeletTransform(min_shapelet_length=min_shp_length, max_shapelet_length=np.inf)
		
		start = time.time()
		
		stc.fit(X_train_sktime, y_train)
		X_train_transformed = stc.transform(X_train_sktime)
		clf.fit(X_train_transformed, y_train)
		X_test_transformed = stc.transform(X_test_sktime)
		clf.score(X_test_transformed, y_test)

		stc_time = time.time() - start

		#  Write results
		result = [nb_series, sast_time, sast_fixed_time, saste_time, saste_fixed_time, stc_time]
		f.write(','.join(np.array(result, dtype=np.str)) + '\n')

print('All done')
