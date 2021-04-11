import numpy as np
import os
import pickle
from utils import load_config_file
import sys


if __name__ == '__main__':
    config_file_name = sys.argv[1]
    config = load_config_file(config_file_name)

    # create output folder if it doesn't exist
    output_folder = config["Output Folder"]
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # load test set
    input_feature_path_testing = config["Input Feature Path Testing"]
    X_testing = np.load(input_feature_path_testing)
    input_age_path_testing = config["Input Age Path Testing"]
    Y_testing = np.load(input_age_path_testing)
    input_subjects_path_testing = config["Input Subject IDs Path Testing"]
    IDs_testing = np.load(input_subjects_path_testing)

    # extract PCs of test set
    pretrained_folder = config["Pretrained Folder"]
    with open(os.path.join(pretrained_folder, 'pca_training_set.pickle'), 'rb') as input_file:
        pca = pickle.load(input_file)
    X_testing = pca.transform(X_testing)
    age_mean = np.load(os.path.join(pretrained_folder, 'mean_age_training_set.npy'))
    Y_training_orig = np.load(os.path.join(pretrained_folder, 'y_training_set.npy'))
    Y_training = Y_training_orig - age_mean

    # age estimation with GPR
    with open(os.path.join(pretrained_folder, 'gpr_model.pickle'), 'rb') as input_file:
        gpr = pickle.load(input_file)
    Y_pred_testing_gpr = gpr.predict(X_testing) + age_mean

    # age estimation with LR if specified
    perform_lr = config["Add LR predictions"]
    if perform_lr:
        with open(os.path.join(pretrained_folder, 'lr_model.pickle'), 'rb') as input_file:
            lr = pickle.load(input_file)
        bin_means, bin_edges = np.histogram(Y_training_orig, 40)
        ix_bins = np.digitize(Y_training_orig, bin_edges)
        Y_training_orig_binned = bin_edges[ix_bins - 1]
        age_mean = np.mean(Y_training_orig_binned)
        Y_training = Y_training_orig_binned - age_mean
        for i in range(np.shape(Y_training)[0]):
            Y_training[i] = int(Y_training[i])
        Y_pred_testing_lr = lr.predict(X_testing) + int(age_mean)

        # get LR weights
        lr_probabilities = lr.predict_proba(X_testing)
        all_ages = np.unique(Y_training) + int(age_mean)
        lr_weights = []
        for i in range(np.shape(lr_probabilities)[0]):
            pred_age_i = Y_pred_testing_lr[i]
            ix = np.where(all_ages == pred_age_i)
            weight_i = lr_probabilities[i][ix][0]
            lr_weights.append(weight_i)

        # weighted average of the predictions
        Y_pred_testing = []
        for i in range(len(lr_weights)):
            w_i = lr_weights[i]
            pred_i = (Y_pred_testing_gpr[i] + w_i * Y_pred_testing_lr[i]) / (1 + w_i)
            Y_pred_testing.append(pred_i)

    else:
        Y_pred_testing = Y_pred_testing_gpr

    # write results in output text file
    text = ''
    curr_ID = ''
    for i in range(len(IDs_testing)):
        test_ID = IDs_testing[i]
        if not curr_ID == test_ID:
            text += test_ID + '\n'
            curr_ID = test_ID
        text += str(Y_pred_testing[i]) + \
                ' - AE = ' + str(np.abs(Y_pred_testing[i] - Y_testing[i])) + \
                ' - Real age = ' + str(Y_testing[i]) + '\n'
    output_file_name = os.path.join(output_folder, 'predictions_test_dataset.txt')
    with open(output_file_name, 'w') as f:
        f.write(text)
    f.close()
