import numpy as np
from sklearn.decomposition import PCA
import os
from sklearn.linear_model import LogisticRegression
import pickle
from utils import load_config_file
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


if __name__ == '__main__':
    config_file_name = sys.argv[1]
    config = load_config_file(config_file_name)

    # create output folder
    output_folder = config["Output Folder"]
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # load training set
    input_feature_path_training = config["Input Feature Path Training"]
    X_training = np.load(input_feature_path_training)
    input_age_path_training = config["Input Age Path Training"]
    Y_training = np.load(input_age_path_training)

    # perform PCA
    pca = PCA(n_components=0.95)
    X_training = pca.fit_transform(X_training)
    with open(os.path.join(output_folder, 'pca_training_set.pickle'), 'wb') as output_file:
        pickle.dump(pca, output_file)

    # normalise ages by subtracting the mean
    age_mean = np.mean(Y_training)
    Y_training_orig = Y_training
    Y_training = Y_training - age_mean
    np.save(os.path.join(output_folder, 'mean_age_training_set.npy'), age_mean)
    np.save(os.path.join(output_folder, 'y_training_set.npy'), Y_training_orig)

    # training GPR
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X_training, Y_training)
    with open(os.path.join(output_folder, 'gpr_model.pickle'), 'wb') as output_file:
        pickle.dump(gpr, output_file)

    perform_lr = config["Add LR classifier"]
    if perform_lr:
        # bin age data for LR classification
        bin_means, bin_edges = np.histogram(Y_training_orig, 40)
        ix_bins = np.digitize(Y_training_orig, bin_edges)
        Y_training_orig_binned = bin_edges[ix_bins - 1]
        age_mean = np.mean(Y_training_orig_binned)
        Y_training = Y_training_orig_binned - age_mean
        for i in range(np.shape(Y_training)[0]):
            Y_training[i] = int(Y_training[i])

        # fit model
        lr = LogisticRegression(solver='newton-cg', class_weight='balanced')
        lr.fit(X_training, Y_training)
        with open(os.path.join(output_folder, 'lr_model.pickle'), 'wb') as output_file:
            pickle.dump(lr, output_file)
