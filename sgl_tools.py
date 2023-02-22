import numpy as np
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state

def poly_data(groups, X):
    import itertools
    group = np.unique(groups)
    print("---Start polynomial features---")
    print("Data matrix:")
    print(X)
    print("Shape of data matrix:")
    print(X.shape)
    print("Group indices:")
    print(groups)
    n_origin_features = X.shape[1]
    print("Number of original features:")
    print(n_origin_features)

    interaction_groups = list(itertools.combinations(group, 2)) # 2 interactions
    groups = np.array([], dtype=np.int32)
    additional_label = 0
    for interaction_group in interaction_groups:
        X = np.append(X, np.ones((X[:][:,interaction_group[0]].shape[0],1)), axis=1)
        X = np.append(X, X[:][:,[interaction_group[0]]], axis=1)
        X = np.append(X, X[:][:,[interaction_group[1]]], axis=1)
        X = np.append(X, np.power(X[:][:,[interaction_group[0]]],2), axis=1)
        X = np.append(X, np.power(X[:][:,[interaction_group[1]]],2), axis=1)
        X = np.append(X, X[:][:,[interaction_group[0]]]*X[:][:,[interaction_group[1]]], axis=1)
        for _ in range(6): groups = np.append(groups, additional_label)
        additional_label += 1
    #return groups, X # poly + origin
    print("Data matrix of polynomial features:")
    print(X.shape)
    print("Data matrix of polynomial features without original features:")
    print(X[:][:,n_origin_features:].shape)
    print(X[:][:,n_origin_features:])
    print("---End polynomial features---")
    return groups, X[:][:,n_origin_features:]

def poly_data_list(groups, X):
    import itertools
    group = np.unique(groups)
    print("---Start polynomial features---")
    print("Data matrix:")
    print(X)
    print("Shape of data matrix:")
    print(X.shape)
    print("Group indices:")
    print(groups)
    n_origin_features = X.shape[1]
    print("Number of original features:")
    print(n_origin_features)

    interaction_groups = list(itertools.combinations(group, 2)) # 2 interactions
    print("Combination of group indices:")
    print(interaction_groups)
    groups = np.array([], dtype=np.int32)
    additional_label = 0

    X_list = []
    for interaction_group in interaction_groups:
        X_list.append(np.ones((X[:][:,interaction_group[0]].shape[0],1)).tolist())
        X_list.append(X[:][:,[interaction_group[0]]].tolist())
        X_list.append(X[:][:,[interaction_group[1]]].tolist())
        X_list.append(np.power(X[:][:,[interaction_group[0]]],2))
        X_list.append(np.power(X[:][:,[interaction_group[1]]],2))
        X_list.append(X[:][:,[interaction_group[0]]]*X[:][:,[interaction_group[1]]].tolist())
        for _ in range(6): groups = np.append(groups, additional_label)
        additional_label += 1
    print("---End polynomial features---")
    return groups, np.asarray(X_list)[:,:,0].T

def build_reg_consts(X, y, group_to_feat, n_lambdas=100, delta=3):
    k = np.dot(X.T, y) # (p*n)*n=p
    lambda_max = 0.0
    n_samples = X.shape[0]
    for feat in group_to_feat:
        k_l2 = np.linalg.norm(k[feat])/n_samples
        if k_l2 > lambda_max: lambda_max = k_l2
    print("Lambda MAX:")
    print(lambda_max)
    lambdas = lambda_max * \
        10 ** (-delta * np.arange(n_lambdas) / (n_lambdas - 1.))
    return lambdas

def reg_const(X, y, group_to_feat, factor=0.01):
    k = np.dot(X.T, y) # (p*n)*n=p
    lambda_max = 0.0
    n_samples = X.shape[0]
    for feat in group_to_feat:
        k_l2 = np.linalg.norm(k[feat])/n_samples
        if k_l2 > lambda_max: lambda_max = k_l2
    print("Lambda MAX:")
    print(lambda_max)
    _lambda = lambda_max * factor
    return np.array([_lambda])
