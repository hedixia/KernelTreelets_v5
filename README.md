# Kernel Treelets (KT)

Kernel Treelets is a hierarchical clustering algorithm proposed by Hedi Xia and Hector D. Ceniceros. 
It combines treelets, a particular multiscale decomposition of data, with a projection on a reproducing kernel Hilbert space. 
The proposed approach, called kernel treelets (KT), effectively substitutes the correlation coefficient matrix used in treelets with a symmetric, 
positive semi-definite matrix efficiently constructed from a kernel function. 
Unlike most clustering methods, which require data sets to be numeric, 
KT can be applied to more general data and yield a multi-resolution sequence of basis on the data directly in feature space. 
A more detailed explanation about Kernel Treelets can be found in arXiv (https://arxiv.org/abs/1812.04808). 
A copy of PDF of this paper is within this repo. The paper is submitted to Advances in Data Science and Adaptive Analysis.
The most recent test version is at https://github.com/hedixia/KernelTreelets_v5.

This repo is only for testing the examples in the paper above. 
The implementation may not be fully optimized and it is still under construction as the setup files are not yet finished. 
Currently it can be used by copying all files in the main folder with .py extensions to the path of the executed file. 

For more infomation about Kernel Method, see (https://en.wikipedia.org/wiki/Kernel_method).

For more infomation about Treelets by Ann B. Lee, Boaz Nadler and Larry Wasserman, see arXiv (https://arxiv.org/abs/0707.0481).
