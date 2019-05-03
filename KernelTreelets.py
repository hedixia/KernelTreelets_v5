import inspect
import itertools

import numpy as np
import sklearn
from sklearn.svm import SVC

SVCkeys = inspect.signature(SVC.__init__).parameters.keys()


def jacobi_rotation (M, k, l, tol=10 ** (-11)):
	"""
	input: numpy matrix for rotation M, two different row number k and l
	output: cos and sin value of rotation
	change: M is inplace changed
	"""

	# rotation matrix calc
	if M[k, l] + M[l, k] < tol:
		cos_val = 1
		sin_val = 0
	else:
		b = (M[l, l] - M[k, k]) / (M[k, l] + M[l, k])
		tan_val = (1 if b >= 0 else -1) / (abs(b) + np.sqrt(b * b + 1))  # |tan_val| < 1
		cos_val = 1 / (np.sqrt(tan_val * tan_val + 1))  # cos_val > 0
		sin_val = cos_val * tan_val  # |cos_val| > |sin_val|

	# right multiplication by jacobian matrix
	temp1 = M[k, :] * cos_val - M[l, :] * sin_val
	temp2 = M[k, :] * sin_val + M[l, :] * cos_val
	M[k, :] = temp1
	M[l, :] = temp2

	# left multiplication by jacobian matrix transpose
	temp1 = M[:, k] * cos_val - M[:, l] * sin_val
	temp2 = M[:, k] * sin_val + M[:, l] * cos_val
	M[:, k] = temp1
	M[:, l] = temp2

	return cos_val, sin_val


class treelets:
	def __init__ (self, verbose=False):
		self.shape = (0,)
		self.verbose = verbose
		self.M_ = None
		self.max_row = None
		self.root = None
		self.dfrk = None
		self._tree = None
		self._layer = None
		self.active = None
		self.active_list = None
		self.transform_list = []
		self.dendrogram_list = []

	def __len__ (self):
		return self.shape[0]

	## Treelet Tree
	@property
	def tree (self):
		if self._tree is None:
			self._tree = [I[0:2] for I in self.transform_list]
		return self._tree

	@property
	def layer (self):
		if self._layer is None:
			self._layer = np.ones(len(self), dtype=int)
			for merging in self.tree:
				self._layer[merging[0]] += self._layer[merging[1]]
		return self._layer

	def fit (self, X):
		self.M_ = np.asarray(X)
		self.shape = self.M_.shape
		self.active = np.ones(len(self), dtype=bool)
		self.max_row = np.zeros(len(self), dtype=int)
		self._rotate(len(self) - 1)
		self.root = self.max_row[np.nonzero(self.active)[0][0]]

	def _rotate (self, multi=False):
		if multi:
			for i in range(multi):
				self._rotate()
				if self.verbose:
					print("rotation: ", i, "\tcurrent: ", self.current)
			self.dfrk = [self.transform_list[i][1] for i in range(len(self) - 1)]
			self.dfrk.append(self.transform_list[-1][0])
		else:
			(p, q) = self._find()
			(cos_val, sin_val) = jacobi_rotation(self.M_, p, q)
			self._record(p, q, cos_val, sin_val)

	def _find (self):
		self.active_list = np.nonzero(self.active)[0]
		if self.transform_list:
			k, l, *_ = self.current
			for i in self.active_list:
				self._maintainance(i, k, l)
		else:
			self.max_row_val = np.zeros(len(self))
			for i in self.active_list:
				self._max(i)

		k = np.argmax(self.max_row_val * self.active)
		v = self.max_row_val[k]
		self.dendrogram_list.append(np.log(v))
		return self.max_row[k], k

	def _maintainance (self, i, k, l):
		if i in (k, l):
			self._max(i)
		if self.M_[self.max_row[i], i] < self.M_[l, i]:
			self.max_row[i] = l
		if self.M_[self.max_row[i], i] < self.M_[k, i]:
			self.max_row[i] = k
		if self.max_row[i] in (k, l):
			self._max(i)

	def _max (self, col_num):
		temp = np.abs(self.M_[col_num]) * (self.active - 0.5)
		temp[col_num] = -1
		self.max_row[col_num] = np.argmax(temp)
		self.max_row_val[col_num] = self.M_[self.max_row[col_num], col_num]

	def _record (self, l, k, cos_val, sin_val):
		if self.M_[l, l] < self.M_[k, k]:
			self.current = (k, l, cos_val, sin_val)
		else:
			self.current = (l, k, cos_val, sin_val)

		self.transform_list.append(self.current)
		self.active[self.current[1]] = False


class kernel_treelets:
	def __init__ (self, kernel=False, verbose=False, **kwargs):
		# Input Variables
		self.kernel_name = kernel if type(kernel) is str else kernel.__class__.__name__
		self._kernel = self._input_kernel(kernel)
		self.__dict__.update(kwargs)
		self.coef_dict = kwargs

		# Intermediate Variables
		self._trl = treelets(verbose=verbose)

		# Output Variables Initialization
		self.shape = None
		self.__X = None
		self.A_0 = None
		self.A_k = None
		self.L_k = None
		self.Delta_k = None
		self._children_list = False

	def __len__ (self):
		return len(self._trl)

	@property
	def transform_list (self):
		return self._trl.transform_list

	@property
	def children_ (self):
		if self._children_list:
			return self._children_list
		else:
			repl = list(range(len(self)))
			children_list = []
			for i in range(len(self.transform_list)):
				children_list.append((repl[self.transform_list[i][0]], repl[self.transform_list[i][1]]))
				repl[self.transform_list[i][0]] = i + len(self)
			self._children_list = children_list
			return children_list

	def fit (self, X, k=-1):
		self.__X = np.asmatrix(X)
		self.shape = self.__X.shape
		n = self.__X.shape[0]
		k = (k + n if k < 0 else k)
		if self._kernel_matrix_function(0, 0):
			A_0 = np.fromfunction(self._kernel_matrix_function, shape=(n, n), dtype=int)
		else:
			A_0 = self._kernel(self.__X)
		self.A_0 = np.matrix(A_0)
		self._trl.fit(A_0)
		A_k = self.transform(self.transform(self.A_0.getT(), k).getT(), k)
		self.A_k = A_k

	def transform (self, v, k=1):
		v = np.matrix(v)
		for i in range(len(self) - k):
			(scv, cgs, cos_val, sin_val) = self.transform_list[i]
			temp_scv = cos_val * v[:, scv] - sin_val * v[:, cgs]
			temp_cgs = sin_val * v[:, scv] + cos_val * v[:, cgs]
			v[:, scv] = temp_scv
			v[:, cgs] = temp_cgs
		return v

	def _decomp (self, M):
		M = np.asmatrix(M)
		farray = [I[1] for I in self.transform_list] + [self._trl.root]

		# rearrange the matrix with the order of tree
		tempf = lambda x, y: M[farray[x], farray[y]]
		M = np.asmatrix(np.fromfunction(np.vectorize(tempf), shape=M.shape, dtype=int))

		# cholesky decomposition
		L = np.sqrt(np.diag(np.diag(M)))

		# rearrange the matrix back to origional order
		barray = np.zeros(len(farray), dtype=int)
		for i in range(len(farray)):
			barray[farray[i]] = i
		L = L[barray, :]
		return L

	def _kernel_matrix_function (self, x, y):
		try:
			return self._kernel(self.__X[x, :], self.__X[y, :])
		except TypeError:
			return False

	def _input_kernel (self, kernel):  # return a kernel function f:SxS->R
		if kernel == "rbf":
			kernel = self._rbf
		if kernel == "poly":
			kernel = self._poly
		if kernel == "linear":
			kernel = self._linear
		return kernel

	# Kernel Handeling
	@property
	def gamma (self):
		if hasattr(self, '_gamma_'):
			return self._gamma_
		if hasattr(self, 'sigma'):
			self._gamma_ = 1 / 2 / self.sigma / self.sigma
			return self._gamma_

	@staticmethod
	def _linear (x, y):  # Linear Kernel
		return (np.asarray(x) * np.asarray(y)).sum(axis=-1)

	def _rbf (self, X):  # Radial Basis Function Kernel
		diff = sklearn.metrics.pairwise.euclidean_distances(X)
		return np.exp(- diff * diff * self.gamma)

	def _poly (self, X):  # Polynomial Kernel
		return (np.inner(X, X) * self.gamma + self.coef0) ** self.degree


class kernel_treelets_SVC(kernel_treelets):
	def __init__ (self, kernel=False, number_of_clusters=0, max_sample=500, label_type=None, verbose=False, **kwargs):
		super().__init__(kernel, verbose=verbose, **kwargs)
		self.max_sample = max_sample
		self.tiny_cluster_number = 0
		self.label_type = label_type

		if number_of_clusters is 0:  # Auto-find clust num
			self.number_of_clusters = 2
			self.clustnum_estimate = True
		else:
			self.number_of_clusters = number_of_clusters
			self.clustnum_estimate = False

		# Output Variables Initialization
		self.dataset = None
		self.samp_dataset = None
		self.tree = None
		self.sample_index = None
		self.sample_labels = None
		self._labels_ = None
		self.svm = None
		self.raw_dataset = None

	@property
	def labels_ (self):
		if self.label_type is None:
			pass
		elif self.label_type == int:
			temp_dict = {v: k for k, v in enumerate(np.unique(self._labels_), 0)}
			self._labels_ = np.vectorize(temp_dict.__getitem__)(self._labels_)
		else:
			temp_dict = {v: k for k, v in enumerate(np.unique(self._labels_), 0)}
			temp_list = itertools.islice(self.label_type, len(temp_dict))
			tempf = lambda x: temp_list[temp_dict[x]]
			self._labels_ = np.apply_along_axis(np.vectorize(tempf), 0, self._labels_)

		return self._labels_

	def fit (self, X, k=-1):
		X = np.asmatrix(X)
		if X.shape[0] <= self.max_sample:  # small dataset
			self.dataset = X
			super().fit(self.dataset, k)

			# clustering on dataset
			self.tree = self._trl.tree
			if self.clustnum_estimate:
				self.find_clust_num(self._trl.dendrogram_list)

			self._labels_ = self.get_labels()
			self.sample_labels = np.array(self._labels_, copy=True)

		else:  # large dataset
			self.raw_dataset = X  # origional copy

			# draw a small sample
			self.sample_index = np.sort(np.random.choice(self.raw_dataset.shape[0], self.max_sample, replace=False))
			self.dataset = self.raw_dataset[self.sample_index, :]

			# build model on small sample
			self.fit(self.dataset, k)
			coef_dict = {key: self.coef_dict[key] for key in self.coef_dict if key in SVCkeys}

			temp = np.unique(self._labels_, return_counts=True)
			valid_labels = np.isin(self._labels_, temp[0][np.argsort(temp[1])][-self.number_of_clusters:])

			self.dataset = self.dataset[valid_labels]
			self._labels_ = self._labels_[valid_labels]

			# generalize to large sample with SVM
			try:
				self.svm = SVC(kernel=self.kernel_name, **coef_dict)
				self.svm.fit(self.dataset, self._labels_)
			except ValueError:
				self.svm = SVC()
				self.svm.fit(self.dataset, self._labels_)
			self._labels_ = self.svm.predict(self.raw_dataset)

	def find_clust_num (self, dendrogram_list):
		# find the first gap with 1
		for i in range(1, len(self) - 1):
			if np.abs(dendrogram_list[i - 1] - dendrogram_list[i]) > 1:
				self.number_of_clusters = len(self) - i
				return self.number_of_clusters

	def get_labels (self):
		temp_labels = np.arange(len(self))
		for i in range(len(self) - self.number_of_clusters - 1, -1, -1):
			temp_labels[self.tree[i][1]] = temp_labels[self.tree[i][0]]
		return temp_labels
