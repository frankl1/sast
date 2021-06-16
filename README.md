# SAST: Scalable and Accurate Subsequence Transform for Time Series Classification

SAST is a novel shapelet-based time series classification method inspired by the *core object recognition* capability of human brain. SAST is more accurate than STC while being more scalable.



SASTEN is an ensemble of 3 SAST models. SASTEN is more accurate than SAST and more scalable than STC.



SASTEN-A is an ensemble of 3 approximated SAST models. The approximation is done by considering only a subset of the subsequences in the dataset.



STC-k is a shapelet transform classifier which generate shapelet candidates from at most *k* reference time series per class. If *k* is a float, then *k x n_c* instances are used per class, where *n_c* is the total number of instances in class *c*.



### Results

- [All models results](./results/all-model-acc.csv)
- [SAST results](./results/results-rf-ridge.csv)
- [STC-k results](./results/results-stc-k.csv)
- [Approximated SAST results: SAST-A](./results/results-sast-approx.csv)
- [Ensemble of approximated SAST results: SASTEN-A](./results/results-sast-ensemble-approx.csv)
- [Ensemble of SAST results: SASTEN](./results/results-sast-ensemble-full.csv)
- [Execution time regarding the number of series](./results/results-scalability-number-of-series.csv)
- [Execution time regarding series length](./results/results-scalability-series-length.csv)



### SAST vs SASTEN

#### Pairwise accuracy comparison

| ![](images/scatter-sast-ridge-A-vs-sast-ridge.jpg) | ![](images/scatter-sasten-ridge-A-vs-sast-ridge.jpg) |
| -------------------------------------------------- | ---------------------------------------------------- |
| ![](images/scatter-sast-rf-vs-ridge.jpg)           | ![](images/scatter-sasten-vs-sast-ridge.jpg)         |

#### Critical difference diagram

![SAST-models CDD](images/cd-sast-models.jpg)

### STC-k vs STC

#### Pairwise accuracy comparison

| ![STC vs STC-1](images/scatter-stc-vs-stck1.png)    | ![STC vs STC-0.25](images/scatter-stc-vs-stck025.png) |
| --------------------------------------------------- | ----------------------------------------------------- |
| ![STC vs STC-0.5](images/scatter-stc-vs-stck05.png) | ![STC vs STC-0.75](images/scatter-stc-vs-stck075.png) |

#### Critical difference diagram

![SCT vs STC-k CDD](images/cdd-stck.png)

### SAST vs STC

| ![SAST vs STC-1](images/scatter-sast-stc1.png) | ![SAST vs STC-1](images/scatter-sast-stc.png) |
| ---------------------------------------------- | --------------------------------------------- |

#### Critical difference diagram

![CDD SAST vs STC](images/cdd-sast-stck.png)

#### Percentage of wins per problem type

![win-per-dataset-type-stck](images/win-per-dataset-type-stck.png)

### SAST vs others shapelets methods

#### Pairwise accuracy comparison

| ![SAST vs ELIS++](images/scatter-sast-elis++.png) | ![SAST vs LS](images/scatter-sast-ls.png) |
| ------------------------------------------------- | ----------------------------------------- |

![SAST vs FS](images/scatter-sast-fs.png)

#### Critical difference diagram

![SAST vs other shapelets CDD](images/cdd-sast-vs-others-shapelet.png)

#### Percentage of wins per problem types

![win-per-dataset-type-shapelet](images/win-per-dataset-type-shapelet.png)

### SAST vs SOTA

#### Pairwise accuracy comparison

| ![scatter-sast-vs-rocket](images/scatter-sast-vs-rocket.jpg) | ![scatter-sast-ridge-vs-hive-cote](images/scatter-sast-ridge-vs-hive-cote.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

![scatter-sast-ridge-vs-chief](images/scatter-sast-ridge-vs-chief.jpg)

#### Percentage of wins per problem type

![win-per-dataset-type-sota](./images/win-per-dataset-type-sota.png)



### Scalability plots

- Regarding the length of time series

![](images/line-scalability-series-length.jpg)

- Regarding the number of time series in the dataset

![](images/line-scalability-nb-series.jpg)

### Comparison to SOTA

![](images/cd-all-models.jpg)

## Usage

```python
import numpy as np
from sast.utils import *
from sast.sast import *
from sklearn.linear_model import RidgeClassifierCV

clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
sast_ridge = SAST(cand_length_list=np.arange(min_shp_length, max_shp_length+1),
		          nb_inst_per_class=nb_inst_per_class, 
		          random_state=None, classifier=clf)

sast_ridge.fit(X_train, y_train)

prediction = sast_ridge.predict(X_test)
```

### Dependencies

- numpy == 1.18.5
- numba == 0.50.1
- scikit-learn == 0.23.1
- sktime == 0.5.3

