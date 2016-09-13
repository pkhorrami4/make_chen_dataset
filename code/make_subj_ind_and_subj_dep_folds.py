import argparse
import os
import sys
import numpy
from sklearn.cross_validation import LeaveOneLabelOut


def get_folds_indices_lolo(y):
    lolo = LeaveOneLabelOut(y)
    fold_inds = []

    for train_split, test_split in lolo:
        print train_split.shape, test_split.shape
        fold_inds.append((train_split, test_split))

    return fold_inds


def save_fold_inds(save_path, fold_inds):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    numpy.save(os.path.join(save_path, 'fold_inds.npy'), fold_inds)
    

def parse_args():
    parser = argparse.ArgumentParser(description='Generate subject independent'
                                                  ' and subject dependent '
                                                  ' folds.')
    parser.add_argument('--input_path', dest='input_path',
                        default='/data/Expr_Recog/Chen_Huang_avdata_python/npy_files/indiv/npy_files_clean/',
                        help='Location of labels (y).')
    parser.add_argument('--save_path', dest='save_path',
                        default='./Temp',
                        help='Folder to save output merged .npy files.')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    input_path = args.input_path
    save_path = args.save_path

    # Load labels
    y = numpy.load(os.path.join(input_path, 'y.npy'))

    print '\nGenerating Subject Independent Folds.'
    subj_indep_fold_inds = get_folds_indices_lolo(y[0, :])
    #print subj_indep_fold_inds

    # Save subject independent fold indices
    subj_indep_save_path = os.path.join(save_path, 'folds', 'subj_ind')
    save_fold_inds(subj_indep_save_path, subj_indep_fold_inds)

    print '\nGenerating Subject Dependent Folds.'
    relative_clip_inds = numpy.mod(y[1, :].astype('int'), 3)
    subj_dep_fold_inds = get_folds_indices_lolo(relative_clip_inds)
    #print subj_dep_fold_inds

    # Save subject dependent fold indices
    subj_dep_save_path = os.path.join(save_path, 'folds', 'subj_dep')
    save_fold_inds(subj_dep_save_path, subj_dep_fold_inds)

    print '\nDone!'
