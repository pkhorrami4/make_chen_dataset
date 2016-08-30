import argparse
import os
import sys
from time import time

import numpy
import skimage.transform

import dlib
from ffvideo import VideoStream


def detect_crop_all_faces(X):
    num_frames = X.shape[0]
    all_cropped_faces = numpy.zeros((num_frames, 3, 96, 96), dtype=numpy.uint8)
    fail_vec = numpy.zeros(num_frames, dtype=numpy.uint8)
    print all_cropped_faces.shape

    for i in range(num_frames):
    # for i in range(100):
        img = X[i, :, :, :]

        # Detect face / landmarks with dlib
        time_start = time()
        detect_flag, landmarks = detect_face_dlib(img)

        # If face detected:
        if detect_flag != 0:
            # Crop it (using landmarks) and convert to grayscale
            crop_frame, bb = crop_frame_using_landmarks(img, landmarks)
            crop_frame = skimage.transform.resize(crop_frame, (96, 96))
            crop_frame = numpy.uint8(crop_frame*255.0)
            # skimage.io.imsave('./img_%.4d.jpg' % i, crop_frame)
            all_cropped_faces[i, :, :, :] = crop_frame.transpose(2, 0, 1)
            fail_vec[i] = 0
            time_elapsed = time() - time_start
            print 'Processing frame (%d/%d) -- %.2f sec.' % (i, num_frames,
                                                             time_elapsed)
        else:
            print 'Face missed in frame (%d/%d)' % (i, num_frames)
            fail_vec[i] = 1

    return all_cropped_faces, fail_vec


def detect_face_dlib(frame):
    num_landmarks = 68
    predictor_path = '/var/research/Code/dlib-18.17/python_examples/' \
                     'shape_predictor/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    det = detector(frame, 1)

    if det:
        det = det[0]
        detect_flag = 1
        landmarks = []
        shape = predictor(frame, det)
        for i in range(num_landmarks):
            part = shape.part(i)
            landmarks.append(part.x)
            landmarks.append(part.y)
        landmarks = numpy.array(landmarks, dtype='float32')
    else:
        detect_flag = 0
        landmarks = numpy.zeros((2*num_landmarks), dtype='float32')

    # print detect_flag, landmarks

    return detect_flag, landmarks


def crop_frame_using_landmarks(frame, landmarks):
    """ Function to crop the face using the detected facial
    landmarks (courtesy of dlib)."""

    landmarks = numpy.reshape(landmarks, (2, len(landmarks)/2), 'F')
    min_x = numpy.min(landmarks[0, :])
    min_y = numpy.min(landmarks[1, :])-30  # include more of the brow
    max_x = numpy.max(landmarks[0, :])
    max_y = numpy.max(landmarks[1, :])

    # print min_x, max_x
    # print min_y, max_y
    crop_frame = frame[min_y:max_y, min_x:max_x, :]
    bb = (min_x, min_y, max_x-min_x, max_y-min_y)
    return crop_frame, bb


def save_out_data(save_path, save_filename, data):
    """Save data as .npy file to location given by save_path."""

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file_path = os.path.join(save_path, save_filename)
    numpy.save(save_file_path, data)


def parse_args():
    parser = argparse.ArgumentParser(description='Detect and extract faces in '
                                                 'specified .npy and save it.', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--npy_file_path', dest='npy_file_path',
                        default='/data/Expr_Recog/Chen_Huang_avdata_python/npy_files_raw/',
                        help='Path to .npy file containing un-cropped faces.')
    parser.add_argument('--save_path', dest='save_path',
                        default='./npy_cropped_faces/',
                        help='Folder to save output .npy files.')
    parser.add_argument('--subj_id', dest='subj_id',
                        help='Subject to extract cropped faces.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print 'Args: ', args

    time_start = time()

    npy_file_path = args.npy_file_path
    save_path = args.save_path
    subj_id = args.subj_id

    # Load data
    input_X_filename = 'X_'+subj_id+'.npy'
    X = numpy.load(os.path.join(npy_file_path, input_X_filename))

    # Detect and crop faces
    all_cropped_faces, fail_vec = detect_crop_all_faces(X)

    # Save data to .npy files
    output_X_filename = 'X_'+subj_id+'.npy'
    output_fail_vec_filename = 'fail_vec_'+subj_id+'.npy'
    save_out_data(save_path, output_X_filename, all_cropped_faces)
    save_out_data(save_path, output_fail_vec_filename, fail_vec)

    time_elapsed = time() - time_start
    print 'Total Execution Time: %.2f sec.' % time_elapsed
