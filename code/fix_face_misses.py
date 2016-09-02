from glob import glob
import os
import numpy
import matplotlib.pyplot as plt
import dlib
import skimage.transform


def find_subjs_w_misssed_frames(fail_vec_path):
    fail_vec_file_list = sorted(glob(os.path.join(fail_vec_path, 'fail_vec*')))
    # print fail_vec_file_list
    
    for fail_vec_file in fail_vec_file_list:
        subj = os.path.split(fail_vec_file)[1].strip('.npy').split('_')[2]        
        fail_vec = numpy.load(fail_vec_file)
        num_faces_missed = numpy.sum(fail_vec)
        print 'Subj: %s -- Num Faces Missed: %d' % (subj, num_faces_missed)


def detect_face_dlib(frame):
    num_landmarks = 68
    predictor_path = '/var/research/Code/dlib-18.17/python_examples/'\
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


def normalize_landmarks(landmarks, face_bb, new_img_size):
    """ Function to readjust the detected facial landmarks when
    the face is cropped out of the frame."""

    # Subtract upper left corner of face bounding box
    rep_face_bb = numpy.tile(face_bb[0:2], len(landmarks)/2)
    landmarks -= rep_face_bb

    # Scale x,y coordinates from face_w, face_h to be in 96x96 image
    scale_vec = numpy.tile([new_img_size/face_bb[2], new_img_size/face_bb[3]],
                           len(landmarks)/2)
    landmarks *= scale_vec

    return landmarks


def find_most_recent_success_inds(fail_vec):
    assert not numpy.all(fail_vec == 1), 'No successes present.'
    assert not numpy.all(fail_vec == 0), 'No failures present.'
    assert fail_vec[0] != 1, 'First frame missed. May need to provide better initialization.'
    
    fail_inds = numpy.where(fail_vec == 1)[0]
    success_inds = numpy.where(fail_vec == 0)[0] 
    #print fail_inds, success_inds
    
    closest_success_inds_all = []
    for fail_ind in fail_inds:
        mask = success_inds < fail_ind                
        closest_success_inds_all.append(success_inds[mask][-1])
    
    return numpy.array(closest_success_inds_all)


#
# How to extract faces from missed frames:
# (i) Take the closest frame that had a good detection
# (ii) Take the detected landmarks
# (iii) Use the landmarks to crop the face out in the missed frames
#
# Note: Assumes the missed frames are contiguous. 
#

def fill_in_missed_frames(subj_id, orig_face_path, crop_face_path,
                          save_path, use_orig_frame=True, display=False):
    
    X_orig = numpy.load(os.path.join(orig_face_path, 'X_'+subj_id+'.npy'))
    X_crop = numpy.load(os.path.join(crop_face_path, 'X_'+subj_id+'.npy'))
    landmarks = numpy.load(os.path.join(crop_face_path, 'landmarks_'+subj_id+'.npy'))
    fail_vec = numpy.load(os.path.join(crop_face_path, 'fail_vec_'+subj_id+'.npy'))
    
    print X_orig.shape
    print X_crop.shape
    print 'Num Frames Missed: ', numpy.sum(fail_vec)
    
    fail_inds = numpy.where(fail_vec == 1)[0]
    print 'Indices of missed frames: ', fail_inds
    
    most_recent_success_inds = find_most_recent_success_inds(fail_vec)
    print 'Indices of most recent successful face detection: ', most_recent_success_inds
    
    for fail_ind, most_recent_success_ind in zip(fail_inds, most_recent_success_inds):
        print 'fail_ind: %d' % fail_ind
        print 'most_recent_success_ind: %d' % most_recent_success_ind
        
        # Detect face in image with last good detection
        det_flag, landmarks_good = detect_face_dlib(X_orig[most_recent_success_ind, :, :, :])
        
        # Use landmarks from detection to crop original frame
        #   if original frame is degraded: use previous good frame instead        
        if use_orig_frame:
            orig_frame = X_orig[fail_ind, :]
        else:
            orig_frame = X_orig[most_recent_success_ind, :]
        crop_frame, bb = crop_frame_using_landmarks(orig_frame, landmarks_good)        
        crop_frame_r = skimage.transform.resize(crop_frame, (96, 96))
        crop_frame_r = numpy.uint8(crop_frame_r*255.0)             
        X_crop[fail_ind, :, :, :] = crop_frame_r.transpose(2, 0, 1)
        
        # Adjust the landmarks
        landmarks_good_norm = normalize_landmarks(landmarks_good, bb, 96)
        landmarks[fail_ind, :] = landmarks_good_norm
        
        if display:
            # Display results
            print 'Showing original frame %d' % fail_ind
            plt.imshow(orig_frame)
            plt.show()

            print 'Cropped frame using landmarks from good detection'
            plt.imshow(crop_frame_r)        
            plt.scatter(landmarks[fail_ind, 0::2], landmarks[fail_ind, 1::2])        
            plt.show()

    print 'Saving filled in frames.'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    numpy.save(os.path.join(save_path, 'X_'+subj_id+'.npy'), X_crop)
    numpy.save(os.path.join(save_path, 'landmarks_'+subj_id+'.npy'), landmarks)


if __name__ == "__main__":
    orig_face_path = '/data/Expr_Recog/Chen_Huang_avdata_python_augmented/npy_files/indiv/npy_files_raw/'
    crop_face_path = '/data/Expr_Recog/Chen_Huang_avdata_python_augmented/npy_files/indiv/npy_files_cropped/'
    save_path = './fixed_npy_files/'

    find_subjs_w_misssed_frames(crop_face_path)

    fill_in_missed_frames('07', orig_face_path, crop_face_path,
                          save_path, use_orig_frame=False, display=False)
    fill_in_missed_frames('18', orig_face_path, crop_face_path,
                          save_path, display=False)
    fill_in_missed_frames('81', orig_face_path, crop_face_path,
                          save_path, display=False)

