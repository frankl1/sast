# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.pipeline import Pipeline

from sktime.utils.data_processing import from_2d_array_to_nested
from sktime.transformations.panel.rocket import Rocket

from numba import njit, prange

from .mass2 import *

@njit(fastmath=True)
def znormalize_array(arr):
    m = np.mean(arr)
    s = np.std(arr)
    
    # s[s == 0] = 1 # avoid division by zero if any
    
    return (arr - m) / (s + 1e-8)
    # return arr

@njit(fastmath=False)
def apply_kernel(ts, arr):
    d_best = np.inf # sdist
    m = ts.shape[0]
    kernel = arr[~np.isnan(arr)] # ignore nan

    # profile = mass2(ts, kernel)
    # d_best = np.min(profile)
    
    l = kernel.shape[0]
    for i in range(m - l + 1):
        d = np.sum((znormalize_array(ts[i:i+l]) - kernel)**2)
        if d < d_best:
            d_best = d

    return d_best

@njit(parallel = True, fastmath=True)  
def apply_kernels(X, kernels):
    nbk = len(kernels)
    out = np.zeros((X.shape[0], nbk), dtype=np.float32)
    for i in prange(nbk):
        k = kernels[i]
        for t in range(X.shape[0]):
            ts = X[t]
            out[t][i] = apply_kernel(ts, k)
    return out

class SAST(BaseEstimator, ClassifierMixin):
    
    def __init__(self, cand_length_list, shp_step = 1, nb_inst_per_class = 1, random_state = None, classifier = None):
        super(SAST, self).__init__()
        self.cand_length_list = cand_length_list
        self.shp_step = shp_step
        self.nb_inst_per_class = nb_inst_per_class
        self.kernels_ = None
        self.kernel_orig_ = None # not z-normalized kernels
        self.kernels_generators_ = {}
        self.random_state = np.random.RandomState(random_state) if not isinstance(random_state, np.random.RandomState) else random_state
        
        self.classifier = classifier
    
    def get_params(self, deep=True):
        return {
            'cand_length_list': self.cand_length_list,
            'shp_step': self.shp_step,
            'nb_inst_per_class': self.nb_inst_per_class,
            'classifier': self.classifier
        }

    def init_sast(self, X, y):

        self.cand_length_list = np.array(sorted(self.cand_length_list))

        assert self.cand_length_list.ndim == 1, 'Invalid shapelet length list: required list or tuple, or a 1d numpy array'

        if self.classifier is None:
            self.classifier = RandomForestClassifier(min_impurity_decrease=0.05, max_features=None) 

        classes = np.unique(y)
        self.num_classes = classes.shape[0]
        
        candidates_ts = []
        for c in classes:
            X_c = X[y==c]
            
            # convert to int because if self.nb_inst_per_class is float, the result of np.min() will be float
            cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)
            choosen = self.random_state.permutation(X_c.shape[0])[:cnt]
            candidates_ts.append(X_c[choosen])
            self.kernels_generators_[c] = X_c[choosen]
            
        candidates_ts = np.concatenate(candidates_ts, axis=0)
        
        self.cand_length_list = self.cand_length_list[self.cand_length_list <= X.shape[1]]

        max_shp_length = max(self.cand_length_list)

        n, m = candidates_ts.shape
        
        n_kernels = n * np.sum([m - l + 1 for l in self.cand_length_list])

        self.kernels_ = np.full((n_kernels, max_shp_length), dtype=np.float32, fill_value=np.nan)
        self.kernel_orig_ = []
        
        k = 0
        for shp_length in self.cand_length_list:
            for i in range(candidates_ts.shape[0]):
                for j in range(0, candidates_ts.shape[1] - shp_length + 1, self.shp_step):
                    end = j + shp_length
                    can = np.squeeze(candidates_ts[i][j : end])
                    self.kernel_orig_.append(can)
                    self.kernels_[k, :shp_length] = znormalize_array(can)
                    k += 1
    
    def fit(self, X, y):
        
        X, y = check_X_y(X, y) # check the shape of the data

        self.init_sast(X, y) # randomly choose reference time series and generate kernels

        X_transformed = apply_kernels(X, self.kernels_) # subsequence transform of X

        self.classifier.fit(X_transformed, y) # fit the classifier

        return self

    def predict(self, X):

        check_is_fitted(self) # make sure the classifier is fitted

        X = check_array(X) # validate the shape of X

        X_transformed = apply_kernels(X, self.kernels_) # subsequence transform of X

        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        check_is_fitted(self) # make sure the classifier is fitted

        X = check_array(X) # validate the shape of X

        X_transformed = apply_kernels(X, self.kernels_) # subsequence transform of X

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed)
        return self.classifier.predict_proba(X_transformed)
    
class SASTEnsemble(BaseEstimator, ClassifierMixin):
    
    def __init__(self, cand_length_list, shp_step = 1, nb_inst_per_class = 1, random_state = None, classifier = None, weights = None, n_jobs = None):
        super(SASTEnsemble, self).__init__()
        self.cand_length_list = cand_length_list
        self.shp_step = shp_step
        self.nb_inst_per_class = nb_inst_per_class
        self.classifier = classifier
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.saste = None

        self.weights = weights

        assert isinstance(self.classifier, BaseEstimator)

        self.init_ensemble()

    def init_ensemble(self):
        estimators = []
        for i, candidate_lengths in enumerate(self.cand_length_list):
            clf = clone(self.classifier)
            sast = SAST(cand_length_list=candidate_lengths,
                          nb_inst_per_class=self.nb_inst_per_class, 
                          random_state=self.random_state, 
                          shp_step = self.shp_step,
                          classifier=clf)
            estimators.append((f'sast{i}', sast))
            

        self.saste = VotingClassifier(estimators=estimators, voting='soft', n_jobs=self.n_jobs, weights = self.weights)

    def fit(self, X, y):
        self.saste.fit(X, y)
        return self

    def predict(self, X):
        return self.saste.predict(X)

    def predict_proba(self, X):
        return self.saste.predict_proba(X)

class RocketClassifier:
    def __init__(self, num_kernels=10000, normalise=True, random_state=None, clf=None, lr_clf=True):
        rocket = Rocket(num_kernels=num_kernels, normalise=normalise, random_state=random_state)
        clf = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10)) if clf is None else clf
        self.model = Pipeline(steps=[('rocket', rocket), ('clf', clf)])
        self.lr_clf = lr_clf # False if the classifier has the method predict_proba, otherwise False
        
    def fit(self, X, y):
        self.model.fit(from_2d_array_to_nested(X), y)
        
    def predict(self, X):
        return self.model.predict(from_2d_array_to_nested(X))
    
    def predict_proba(self, X):
        X_df = from_2d_array_to_nested(X)
        if not self.lr_clf:
            return self.model.predict_proba(X_df)
        X_transformed = self.model['rocket'].transform(X_df)
        return self.model['clf']._predict_proba_lr(X_transformed)

if __name__ == "__main__":
    a = np.arange(10, dtype=np.float32).reshape((2, 5))
    y = np.array([0, 1])
    print('input=\n', a)
    print('y=\n', y)
    
    ## SAST
    sast = SAST(cand_length_list=np.arange(2, 5), nb_inst_per_class=2, classifier=RidgeClassifierCV())
    
    sast.fit(a, y)
    
    print('kernel:\n', sast.kernels_)

    print('Proba:', sast.predict_proba(a))

    print('score:', sast.score(a, y))

    ## SASTEnsemble
    saste = SASTEnsemble(cand_length_list=[np.arange(2, 4), np.arange(4, 6)], nb_inst_per_class=2, classifier=RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    
    saste.fit(a, y)

    print('SASTEnsemble Proba:', sast.predict_proba(a))

    print('SASTEnsemble score:', sast.score(a, y))


