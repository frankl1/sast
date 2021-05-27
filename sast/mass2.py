import numpy as np
from numba import njit, guvectorize, objmode

@njit
def moving_mean_std(a, window_size):
	max_len = len(a) - window_size + 1 
	mean = np.zeros(max_len)
	std = np.zeros(max_len)
	for i in range(max_len):
		mean[i] = np.mean(a[i:i+window_size])
		std[i] = np.std(a[i:i+window_size])
	return mean, std

@njit()
def mass2(ts, query, threshold=1e-10):
	"""
	Compute the distance profile for the given query over the given time 
	series.

	Parameters
	----------
	ts : array_like
		The time series to search.
	query : array_like
		The query.

	Returns
	-------
	np.array, dict : distance_profile
		An array of distances np.array() or dict with extras.

	Credit
	------ 
	https://github.com/matrix-profile-foundation/matrixprofile

	@article{Van Benschoten2020,
		doi = {10.21105/joss.02179},
		url = {https://doi.org/10.21105/joss.02179},
		year = {2020},
		publisher = {The Open Journal},
		volume = {5},
		number = {49},
		pages = {2179},
		author = {Andrew Van Benschoten and Austin Ouyang and Francisco Bischoff and Tyler Marrs},
		title = {MPA: a novel cross-language API for time series analysis},
		journal = {Journal of Open Source Software}
	}
	"""

	n = len(ts)
	m = len(query)
	x = ts
	y = query

	meany = np.mean(y)
	sigmay = np.std(y)
	
	meanx, sigmax = moving_mean_std(x, m)
	meanx = np.append(np.ones((1, len(x) - len(meanx))), meanx)	
	sigmax = np.append(np.zeros((1, len(x) - len(sigmax))), sigmax)

	
	y = np.append(np.flip(y), np.zeros((1, n - m)))

	z = None
	with objmode(z='complex128[:]'):
		X = np.fft.fft(x)
		Y = np.fft.fft(y)
		Y.resize(X.shape)
		Z = X * Y
		z = np.fft.ifft(Z)
	
	# do not allow divide by zero
	tmp = (sigmax[m - 1:n] * sigmay)
	tmp[tmp == 0] = 1e-12

	dist = 2 * (m - (z[m - 1:n] - m * meanx[m - 1:n] * meany) / tmp)

	# fix to handle constant values
	dist[sigmax[m - 1:n] < threshold] = m
	dist[(sigmax[m - 1:n] < threshold) & (sigmay < threshold)] = 0
	dist = np.sqrt(dist)

	return np.real(dist)