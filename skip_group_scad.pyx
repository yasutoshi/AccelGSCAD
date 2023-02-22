from libc.math cimport fabs, sqrt
from libc.stdlib cimport qsort
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal, dgemv, dgemm, dasum, dcopy
import numpy as np
cimport numpy as np
cimport cython

from sgl_tools import poly_data, build_reg_consts, reg_const
from sklearn import datasets
import argparse
import scipy.linalg

import math
import numpy as np
from scipy import linalg
import time

MAX_ITER = 100000

def scipy_eigh(X):
    w, v = scipy.linalg.eigh(X,
        overwrite_a=False,
        check_finite=False
    )
    return w, v

cdef inline double fmax(double x, double y) nogil:
    if x > y:
        return x
    return y

cdef inline double fsign(double x) nogil:
    if x == 0:
        return 0
    elif x > 0:
        return 1.0
    else:
        return -1.0

cdef double soft_threshold(double x, double threshold) nogil:
    return fsign(x) * fmax(fabs(x) - threshold, 0)

def light_skip_group_scad(double[::1, :] _X, double[::1] y, int n_samples, int n_features, int[::1] groups, int[::1] group_sizes, int[::1, :] group_to_feat, double[::1] group_scalars, int n_groups, double alpha, double gamma, int n_lambdas=100, double factor=0.01, int max_iter=MAX_ITER, double rtol=1e-5):

    cdef:
        double scalar1 = 1.0
        double scalar2 = 0.0
        double scalar3 = -1.0
        double delta = 1.0e-5
        int inc = 1
        int i, g, _g, f, _f, n_iter, c_iter
        int cnt1, cnt2, cnt3, cnt4, int_tmp      

        double[::1] XTR = np.zeros(n_features, dtype = np.float64, order='F')
        double[::1] XTR_ref = np.zeros(n_features, dtype = np.float64, order='F')
        double[::1] Xy = np.empty(n_features, dtype = np.float64, order='F')
        double[::1,:] K = np.empty((n_features, n_features), dtype = np.float64, order='F')
        double[::1,:] X = np.empty((n_samples, n_features), dtype = np.float64, order='F') # transformed X by eigen decomposition
        double[::1] beta = np.zeros(n_features, dtype = np.float64, order='F')
        double[::1] beta_ref = np.zeros(n_features, dtype = np.float64, order='F')
        double[::1,:] K_norms = np.zeros((n_groups, n_groups), dtype = np.float64, order='F')
        double[::1] beta_diff_norms = np.zeros(n_groups, dtype = np.float64, order='F')
        double[::1] zero_vec = np.zeros(n_groups, dtype = np.float64, order='F')

        double[::1] lambdas = np.empty(n_lambdas, dtype = np.float64, order='F')

        double double_tmp = 0.
        double norm_tmp = 0.
        double group_scalar = 0.
        double c = 0.
        double lamb = 0.
        double[::1] Kbeta = np.empty(n_features, dtype = np.float64, order='F')
        double[::1] beta_old = np.zeros(n_features, dtype = np.float64, order='F')
        double[::1] array_tmp = np.zeros(n_features, dtype = np.float64, order='F')
        double[::1] beta_tmp = np.zeros(n_features, dtype = np.float64, order='F')
        list candidate_set = []
        double[::1,:] Kg = np.empty((6, 6), dtype = np.float64, order='F')
        double[::1,:] Tg = np.empty((6, 6), dtype = np.float64, order='F')

    with nogil:
        
        for g in range(n_groups):
            for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                for _f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                    Kg[f-group_to_feat[g][0], _f-group_to_feat[g][0]] = 0.0
                    for i in range(n_samples):
                        Kg[f-group_to_feat[g][0], _f-group_to_feat[g][0]] += _X[i, f]*_X[i, _f]
                    Kg[f-group_to_feat[g][0], _f-group_to_feat[g][0]] /= n_samples
                    if f == _f:
                        Kg[f-group_to_feat[g][0], _f-group_to_feat[g][0]] += delta

            with gil:
                val, vec = scipy_eigh(Kg)
                Tg = np.asfortranarray(np.matmul(vec, np.diag(np.reciprocal(np.sqrt(val)))))

            for i in range(n_samples):
                for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                    double_tmp = 0.0
                    for _f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                        double_tmp += _X[i,_f]*Tg[_f-group_to_feat[g][0],f-group_to_feat[g][0]]
                    X[i,f] = double_tmp

        dgemv("T", &n_samples, &n_features, &scalar1, &X[0,0], &n_samples, &y[0], &inc, &scalar2, &Xy[0], &inc) 
        dgemm("T", "N", &n_features, &n_features, &n_samples, &scalar1, &X[0,0], &n_samples, &X[0,0], &n_samples, &scalar2, &K[0,0], &n_features)

        # compute norms for each block in gram matrix (for computing bounds)
        for g in range(n_groups):
            for _g in range(n_groups):
                for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                    for _f in range(group_to_feat[_g][0], group_to_feat[_g][0]+group_sizes[_g]):
                        K_norms[g,_g] += K[f,_f]**2
                K_norms[g,_g] = sqrt(K_norms[g,_g])

        # build lambda
        with gil:
            if n_lambdas > 1:
                lambdas = build_reg_consts(X, y, group_to_feat)
            else:
                lambdas = reg_const(X, y, group_to_feat, factor)

            print("Lambdas:")
            print(lambdas)

        for l in range(n_lambdas):
            cnt1 = 0
            cnt2 = 0
            cnt3 = 0
            cnt4 = 0
            lamb = lambdas[l]
            for f in range(n_features): beta[f] = 0.
            for n_iter in range(max_iter):
                dcopy(&n_features, &beta[0], &inc, &beta_old[0], &inc)

                # compute reference vectors
                if n_iter == 0:
                    dcopy(&n_features, &beta[0], &inc, &beta_ref[0], &inc)
                    dcopy(&n_groups, &zero_vec[0], &inc, &beta_diff_norms[0], &inc)
                    dcopy(&n_features, &Xy[0], &inc, &XTR_ref[0], &inc) # XTR_ref = Xy
                    dgemv("N", &n_features, &n_features, &scalar3, &K[0,0], &n_features, &beta[0], &inc, &scalar1, &XTR_ref[0], &inc) # XTR_ref = Xy - K*beta
                    for g in range(n_groups):
                        for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                            double_tmp = 0.
                            for _f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                                double_tmp += K[f, _f]*beta[_f]
                            XTR_ref[f] += double_tmp
                            XTR_ref[f] /= n_samples

                for g in range(n_groups):
                    group_scalar = group_scalars[g]

                    # compute upper bound
                    norm_tmp = 0.
                    for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                        norm_tmp += XTR_ref[f]**2
                    norm_tmp = sqrt(norm_tmp)
                    norm_tmp += (K_norms[g, g]*beta_diff_norms[g] + ddot(&n_groups, &beta_diff_norms[0], &inc, &K_norms[0, g], &inc))/sqrt(n_samples)
                    # skip computation using upper bound
                    if norm_tmp < lamb * group_scalar:
                        cnt1 += 1
                        int_tmp = 0
                        for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                            if beta[f] != 0.0:
                                int_tmp = 1
                                break
                        if int_tmp == 0:
                            continue
 
                        double_tmp = 0.
                        for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                            beta[f] = 0.
                            double_tmp += (beta_ref[f] - beta[f])**2
                            XTR_ref[f] = Xy[f]
                            XTR_ref[f] -= ddot(&n_features, &beta[0], &inc, &K[0,f], &inc)
                            XTR_ref[f] /= n_samples
                            beta_ref[f] = beta[f]
                        beta_diff_norms[g] = sqrt(double_tmp)
                        continue

                    # standard group scad
                    for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                        XTR[f] = Xy[f] # instead of (1)
                        XTR[f] -= ddot(&n_features, &beta[0], &inc, &K[0,f], &inc) # XTR = Xy - K*beta
                    norm_tmp = 0.
                    for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                        double_tmp = 0.
                        for _f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                            double_tmp += K[f, _f]*beta[_f]
                        XTR[f] += double_tmp # XTR = Xy - K*beta + K*beta^(g)
                        XTR[f] /= n_samples 
                        XTR_ref[f] = XTR[f]
                        beta_ref[f] = beta[f]
                        norm_tmp += soft_threshold(XTR[f], lamb * alpha)**2 #now alpha=0, so equals to norm_tmp+=XTR[f]**2
                    norm_tmp = sqrt(norm_tmp)
                    if norm_tmp <= 2.0 * lamb * group_scalar:
                        if norm_tmp != 0.0:
                            double_tmp = soft_threshold(norm_tmp, lamb * group_scalar)/norm_tmp 
                            for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                                beta[f] = double_tmp*XTR[f]
                        else:
                            for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                                beta[f] = 0.0
                    elif norm_tmp > 2.0 * lamb * group_scalar and norm_tmp <= gamma * lamb * group_scalar:
                        cnt2+=1
                        double_tmp =  (gamma-1.0)/(gamma-2.0)*soft_threshold(norm_tmp, gamma * lamb * group_scalar/(gamma-1.0))/norm_tmp
                        for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                            beta[f] = double_tmp*XTR[f]
                    elif norm_tmp > gamma * lamb * group_scalar:
                        cnt3+=1
                        for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                            beta[f] = XTR[f]

                    # update norm of difference between old and new parameters
                    double_tmp = 0.
                    for f in range(group_to_feat[g][0], group_to_feat[g][0]+group_sizes[g]):
                        double_tmp += (beta_ref[f] - beta[f])**2
                    beta_diff_norms[g] = sqrt(double_tmp)

                double_tmp = fmax(dnrm2(&n_features, &beta[0], &inc), 1e-10)
                dcopy(&n_features, &beta[0], &inc, &beta_tmp[0], &inc)
                daxpy(&n_features, &scalar3, &beta_old[0], &inc, &beta_tmp[0], &inc)
                norm_tmp = dnrm2(&n_features, &beta_tmp[0], &inc)
                if norm_tmp / double_tmp < rtol:
                    with gil:
                        print("STOP")
                        print(n_iter)
                    break
            with gil:
                print(cnt1)
                print(cnt2)
                print(cnt3)

    beta_ = np.asarray(beta)
    loss = np.sum(np.square(np.asarray(y)-np.dot(X,beta_)))/2.0/n_samples
    cnt_ = 0
    tmp_ = 0.0
    objective = loss
    for b in beta_:
        tmp_ += b*b
        cnt_ += 1
        if cnt_ == 6:
            if np.sqrt(tmp_) <= lamb:
                objective += lamb*np.sqrt(tmp_)
            elif np.sqrt(tmp_) > lamb and np.sqrt(tmp_) <= gamma*lamb:
                objective += (gamma*lamb*np.sqrt(tmp_) - 0.5*(tmp_+lamb*lamb))/(gamma-1.0)
            elif gamma*lamb < np.sqrt(tmp_):
                objective += (lamb*lamb*(gamma*gamma-1.0))/(2.0*(gamma-1.0))
            else:
                print("ERROR")
            tmp_ = 0.0
            cnt_ = 0
    return np.asarray(beta), loss, objective

