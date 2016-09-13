import argparse
from glob import glob
import os
import sys
import numpy


def load_data_and_labels(input_path):
    X_all = []
    y_all = []

    data_file_list = sorted(glob(os.path.join(input_path, 'X_*.npy')))
    label_file_list = sorted(glob(os.path.join(input_path, 'y_*.npy')))

    for data_file, label_file in zip(data_file_list, label_file_list):
        data_filename = os.path.split(data_file)[1]
        label_filename = os.path.split(data_file)[1]
        print 'Reading: %s and %s' % (data_filename, label_filename)
        X = numpy.load(data_file)
        y = numpy.load(label_file)

        X_all.append(X)
        y_all.append(y)

    print 'Read %d data files and %d label files.' % (len(X_all), len(y_all))

    return X_all, y_all


def add_global_clip_numbering(y_list):
    global_clip_counter = 0

    for i, y in enumerate(y_list):
        # Get each subject's clip numbers (0-32)
        clip_nums = y[1, :]
        clip_nums_int = numpy.array([int(clip_num) for clip_num in clip_nums])
        # print 'Clip nums before: ', numpy.unique(clip_nums_int)

        # Adjust them to the global clip numbers
        clip_nums_int += global_clip_counter
        # print 'Clip nums after: ', numpy.unique(clip_nums_int)

        global_clip_counter = (numpy.max(clip_nums_int)+1)

        # Replace relative clip numbers with global clip numbers
        #y[1, :] = clip_nums_int
        #y_list[i] = y

        # Add global clip numbers
        y_list[i] = numpy.vstack((y, numpy.array(clip_nums_int)[None, :]))


    return y_list


def load_landmarks(input_path):
    landmarks_all = []    
    landmark_file_list = sorted(glob(os.path.join(input_path, 'landmarks_*.npy')))    

    for landmark_file in landmark_file_list:        
        landmark_filename = os.path.split(landmark_file)[1]
        print 'Reading: %s' % landmark_filename
        landmarks = numpy.load(landmark_file)
        
        landmarks_all.append(landmarks)

    print 'Read %d landmark files.' % len(landmarks_all)

    return landmarks_all


def save_data(save_path, X, y):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    numpy.save(os.path.join(save_path, 'X.npy'), X)
    numpy.save(os.path.join(save_path, 'y.npy'), y)


def save_landmarks(save_path, landmarks):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    numpy.save(os.path.join(save_path, 'landmarks.npy'), landmarks)


def parse_args():
    parser = argparse.ArgumentParser(description='Read .npy files of '
                                                 'indivdiual subjects and '
                                                 'merges them into one .npy '
                                                 'file. Clip numbers are also '
                                                 'adjusted to be 0-660 instead'
                                                 'of 0-32 for each subject.')
    parser.add_argument('--input_path', dest='input_path',
                        default='/data/Expr_Recog/Chen_Huang_avdata_python/npy_files/indiv/npy_files_clean/',
                        help='Folder containing individual .npy files for '
                             'data (X) and labels (y).')
    parser.add_argument('--save_path', dest='save_path',
                        default='/data/Expr_Recog/Chen_Huang_avdata_python/npy_files/all/',
                        help='Folder to save output merged .npy files.')
    parser.add_argument('--include_landmarks', dest='include_landmarks',
                        action='store_true', default=False,
                        help='Flag to merge facial landmarks.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    input_path = args.input_path
    save_path = args.save_path
    include_landmarks = args.include_landmarks
    # input_path = '/data/Expr_Recog/Chen_Huang_avdata_python/npy_files/indiv/npy_files_clean/'
    # save_path = '/data/Expr_Recog/Chen_Huang_avdata_python/npy_files/all/'

    # Load the data (X) and labels (y)
    print '\nLoading data and label files.'
    X_all, y_all = load_data_and_labels(input_path)

    # Adjust the clip numbers
    print '\nConverting relative clip numbers to global clip numbering.'
    y_all = add_global_clip_numbering(y_all)

    # Merge data and labels from all subjects and save them
    print '\nMerging all of the subjects.'
    X = numpy.concatenate(X_all, axis=0)
    y = numpy.concatenate(y_all, axis=1)
    print X.shape, y.shape

    # Checks global clip numbering
    # sorted(numpy.unique(y[1, :]))
    print '\nSaving merged data and labels.'
    save_data(save_path, X, y)

    # Load and save landmarks (if desired).
    if include_landmarks:
        print '\nLoading landmark files.'
        landmarks_all = load_landmarks(input_path)
        landmarks = numpy.concatenate(landmarks_all, axis=0)
        print '\nSaving merged landmarks.'
        save_landmarks(save_path, landmarks)

    print '\nDone!'
