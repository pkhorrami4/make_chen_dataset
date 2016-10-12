import argparse
import os
import numpy


def compute_mean_std(X):
    mean_train = numpy.mean(X, axis=0)
    std_train = numpy.std(X, axis=0)

    # print mu_train.shape, std_train.shape
    return mean_train, std_train


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute mean and std of '
        'facial landmark features '
        ' and facial landmark differences '
        'and save them as .npy files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_path',
        dest='dataset_path',
        default='/data/Expr_Recog/Chen_Huang_avdata_python_augmented/npy_files/all/',
        help='Folder containing landmark features.')
    parser.add_argument(
        '--feat_type',
        dest='feat_type',
        choices=['landmarks', 'landmarks_diff'],
        help='Which feature to compute stats.')
    parser.add_argument(
        '--fold_type',
        dest='fold_type',
        choices=['subj_dep', 'subj_ind'],
        help='Use subject dependent or indpendent folds.')
    parser.add_argument(
        '--save_path',
        dest='save_path',
        default='/data/Expr_Recog/Chen_Huang_avdata_python_augmented/npy_files/all/global_stats/',
        help='Folder to save output .npy files.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    dataset_path = args.dataset_path
    save_path = args.save_path
    feat_type = args.feat_type
    fold_type = args.fold_type

    save_path = os.path.join(save_path, feat_type)
    fold_inds = numpy.load(
        os.path.join(dataset_path, 'folds', fold_type, 'fold_inds.npy'))

    if feat_type == 'landmarks':
        data = numpy.load(os.path.join(dataset_path, 'landmarks.npy'))
    elif feat_type == 'landmarks_diff':
        data = numpy.load(os.path.join(dataset_path, 'landmarks_diff.npy'))

    global_stats = {}
    fold = 0

    for train_inds, test_inds in fold_inds:
        print 'Fold %d' % fold
        print 'Train, Test Split sizes: ', train_inds.shape, test_inds.shape

        data_train = data[train_inds, :]
        data_test = data[test_inds, :]
        print 'data_train: %s --- data_test: %s' % (data_train.shape,
                                                    data_test.shape)

        mean_train, std_train = compute_mean_std(data_train)
        # data_train_norm = (data_train - mean_train) / (std_train + 1e-6)
        # print numpy.mean(landmark_diff_train_norm, axis=0)
        # print numpy.std(landmark_diff_train_norm, axis=0)
        # print landmark_diff_train_norm.shape

        global_stats[fold] = {}
        global_stats[fold]['mean'] = mean_train
        global_stats[fold]['std'] = std_train
        fold += 1
        print ''

    print 'Saving to .npy file.'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    numpy.save(
        os.path.join(save_path, 'global_stats_' + fold_type + '.npy'),
        global_stats)
