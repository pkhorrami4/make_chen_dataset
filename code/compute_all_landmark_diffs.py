import argparse
import os
import numpy

from chen_huang.data_batchers.video_seq_generator import VideoSequenceGenerator, FeatureSequenceGenerator


def compute_landmark_diffs_single_clip(landmarks):
    landmarks_diff = landmarks[1:, :] - landmarks[0:landmarks.shape[0] - 1, :]
    landmarks_diff = numpy.vstack((numpy.zeros(
        (1, landmarks.shape[1]), landmarks.dtype), landmarks_diff))
    return landmarks_diff


def compute_landmark_diffs_all(landmarks, y):
    print landmarks.shape
    print y.shape

    global_clip_ids = y[-1, :].astype('int')
    print global_clip_ids

    landmarks_diff_list = []
    for clip_id in range(660):
        if clip_id % 50 == 0:
            print 'Processing Global Clip ID: %d' % clip_id

        clip_inds = numpy.where(global_clip_ids == clip_id)[0]
        landmarks_clip = landmarks[clip_inds, :]
        landmarks_diff_clip = compute_landmark_diffs_single_clip(
            landmarks_clip)
        landmarks_diff_list.append(landmarks_diff_clip)

    landmarks_diff = numpy.vstack(landmarks_diff_list)
    print 'landmarks_diff shape: ', landmarks_diff.shape

    return landmarks_diff


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate differences between '
        'facial landmark features '
        'and save as .npy files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_path',
        dest='dataset_path',
        default='/data/Expr_Recog/Chen_Huang_avdata_python_augmented/npy_files/all/',
        help='Folder containing landmark and label (y) .npy files.')
    parser.add_argument(
        '--save_path',
        dest='save_path',
        default='./',
        help='Folder to save output .npy files.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path
    save_path = args.save_path

    y = numpy.load(os.path.join(dataset_path, 'y.npy'))
    landmarks = numpy.load(os.path.join(dataset_path, 'landmarks.npy'))

    landmarks_diff = compute_landmark_diffs_all(landmarks, y)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    numpy.save(os.path.join(save_path, 'landmarks_diff.npy'), landmarks_diff)
