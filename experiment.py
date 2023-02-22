from contextlib import contextmanager
import numpy as np

from sgl_tools import poly_data, poly_data_list, build_reg_consts
from sklearn import datasets
import argparse
import scipy.linalg

import math
import numpy as np
from scipy import linalg
import time

from sklearn.datasets import fetch_openml

from group_scad import group_scad
from fast_group_scad import fast_group_scad
from skip_group_scad import light_skip_group_scad

@contextmanager
def timer(title):
    t0 = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - t0
        print("{} - done in {:.6f}s".format(title, elapsed_time))
        with open(title+".csv", mode='w') as f:
            f.write(str(elapsed_time))

def experiment():
    parser = argparse.ArgumentParser(description='Experimental code of Group SCAD(group_scad)/Fast Group SCAD(skip_group_scad)/AccelGSCAD(fast_group_scad, ours).')
    parser.add_argument('--method', default='group_scad', choices=['group_scad', 'skip_group_scad', 'fast_group_scad'])
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--gamma', default=3.7, type=float)
    parser.add_argument('--lambda_factor', default=0.01, type=float)
    parser.add_argument('--max_iter', default=1000000, type=int)
    parser.add_argument('--r_tol', default=1e-5, type=float)
    parser.add_argument('--save_poly', action='store_true')
    parser.add_argument('--data', default='eunite', choices=[
        'eunite',
        'triazines',
        'qsbr_rw1',
        'qsf',
        'qsbralks'
    ])
    args = parser.parse_args()

    np.random.seed(0)
    print('Reading args...')
    print(args.method)
    print(args.data)
    print(args.alpha)
    print(args.gamma)
    print(args.lambda_factor)

    # hyper parameters
    alpha_ = args.alpha #alpha is balancing factor, just using group penelty as alpha=0.0 (do not use this code)
    gamma_ = args.gamma
    factor_ = args.lambda_factor
    max_iter_ = args.max_iter
    r_tol_ = args.r_tol

    #origin_data = datasets.fetch_california_housing()
    if args.data=="eunite":
        origin_data = datasets.load_svmlight_file("./data/eunite2001")
        X = origin_data[0].toarray()
        y = origin_data[1]
    elif args.data=="triazines":
        origin_data = datasets.load_svmlight_file("./data/triazines_scale")
        X = origin_data[0].toarray()
        y = origin_data[1]
    elif args.data=="qsbr_rw1":
        origin_data = fetch_openml(data_id=442)
        X = origin_data.data
        y = origin_data.target
    elif args.data=="qsf":
        origin_data = fetch_openml(data_id=427)
        X = origin_data.data
        y = origin_data.target
    elif args.data=="qsbralks":
        origin_data = fetch_openml(data_id=431)
        X = origin_data.data
        y = origin_data.target
    else:
        assert "No dataset."

    print("Shape before polynomial features:")
    print(X.shape)

    # pre-processing
    groups = np.arange(X.shape[1]) // 1
    #groups, X = poly_data_list(groups, X) # faster than np version, but sometimes stop
    groups, X = poly_data(groups, X)
    groups = groups.astype(np.int32)
    if args.save_poly == True:
        np.savetxt(str(args.data) + '_group.csv', groups, delimiter=',')
        np.savetxt(str(args.data) + '_poly.csv', X, delimiter=',')
    print("Shape after polynomial features:")
    print(X.shape)
    n_samples, n_features = X.shape
    n_groups = len(np.unique(groups))
    print("Number of groups:")
    print(n_groups)

    group_to_feat = [np.where(groups == i)[0] for i in np.unique(groups)]

    group_sizes = np.array([group.size for group in group_to_feat]).astype(np.int32)
    print("Group sizes:")
    print(group_sizes)

    group_scalars = np.array([math.sqrt(group.size) for group in group_to_feat])
    print("Group_scalars:")
    print(group_scalars)

    group_to_feat = np.array(group_to_feat).astype(np.int32)
    print("Group_to_feat:")
    print(group_to_feat)

    with timer('time_'+str(args.data)+'_'+str(args.lambda_factor)+'_'+str(args.method)):
        X, y, groups, group_sizes, group_scalars, group_to_feat = map(np.asfortranarray, (X, y, groups, group_sizes, group_scalars, group_to_feat))
        if args.method =='group_scad':
            coefs, loss, obj = group_scad(X, y, n_samples, n_features, groups, group_sizes, group_to_feat, group_scalars, n_groups, alpha_, gamma_, 1, factor_, max_iter_, r_tol_)
        elif args.method =='skip_group_scad':
            coefs, loss, obj = light_skip_group_scad(X, y, n_samples, n_features, groups, group_sizes, group_to_feat, group_scalars, n_groups, alpha_, gamma_, 1, factor_, max_iter_, r_tol_)
        elif args.method =='fast_group_scad':
            coefs, loss, obj = fast_group_scad(X, y, n_samples, n_features, groups, group_sizes, group_to_feat, group_scalars, n_groups, alpha_, gamma_, 1, factor_, max_iter_, r_tol_)
    print('LOSS:')
    print(loss)
    print('OBJ:')
    print(obj)
    print('BETA:')
    print(coefs)
    np.savetxt('loss_'+str(args.data)+'_'+str(args.lambda_factor)+'_'+str(args.method)+'.csv', np.array([loss])) 
    np.savetxt('obj_'+str(args.data)+'_'+str(args.lambda_factor)+'_'+str(args.method)+'.csv', np.array([obj])) 
    np.savetxt('beta_'+str(args.data)+'_'+str(args.lambda_factor)+'_'+str(args.method)+'.csv', coefs) 

if __name__ == '__main__':
    experiment()
