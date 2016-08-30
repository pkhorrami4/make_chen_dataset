import argparse
import os
import sys
import numpy
from ffvideo import VideoStream


def load_label_data_from_file(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    label_data = numpy.vstack([line.strip().split(',')
                               for line in lines]).astype('int')
    print 'Label Data: ', label_data

    return label_data


def calc_frame_boundaries(vs, start, end, scale_const):
    time_start = start / scale_const
    time_end = end / scale_const

    # Convert times to frame numbers
    frame_ind_start = int(numpy.floor(time_start * vs.framerate))
    frame_ind_end = int(numpy.floor(time_end * vs.framerate))
    num_frames_extract = frame_ind_end - frame_ind_start + 1

    print 'Time (sec) -- (Start, End): %.3f, %.3f' % (time_start, time_end)
    print 'Time -- Start (mm:ss): (%.2f, %.2f)' % divmod(time_start, 60)
    print 'Time -- End (mm:ss): (%.2f, %.2f)' % divmod(time_end, 60)    
    print 'Frame Index -- (Start, End): (%d, %d)' % (frame_ind_start, frame_ind_end+1)
    print 'Number of frames to extract: %d' % num_frames_extract

    return frame_ind_start, frame_ind_end, num_frames_extract


def get_clip(vs, start, end, scale_const):
    """
    Load clip from video given the start and end times.
    """
    frame_ind_start, frame_ind_end, num_frames_extract = calc_frame_boundaries(vs,
                                                                               start,
                                                                               end,
                                                                               scale_const)

    all_frames = []
    timestamps = []
    for frame_ind in range(frame_ind_start, frame_ind_end+1):
        video_frame = vs.get_frame_no(frame_ind).ndarray()[numpy.newaxis, :]
        all_frames.append(video_frame)
        timestamps.append(float(frame_ind) / vs.framerate)

    all_frames = numpy.concatenate(all_frames, axis=0)
    print all_frames.shape

    return all_frames, timestamps


def load_data_single_subj(subj_id, video_path, label_path, gender, include_expression_frames=False):
    print '\nProcessing subject %s' % subj_id
    if subj_id == '07':
        video_filename = subj_id+'emotion.avi'
    else:
        video_filename = subj_id+'emotion.00.avi'
    video_file_path = os.path.join(video_path, video_filename)
    label_filename = subj_id+'.label'
    label_file_path = os.path.join(label_path, label_filename)

    vs = VideoStream(video_file_path)
    label_data = load_label_data_from_file(label_file_path)

    num_labels = 11
    num_clips = 3
    num_frames_total = 0

    all_frames = []
    subj_ids = []
    clip_ids = []
    timestamps = []    
    emotion_labels = []
    clip_scale_const = {'F': 92.0, 'M': 90.0}
    if include_expression_frames:
        easiness_labels = []
        clip_extend_amount = {'F': 500, 'M': 500}    

    for i in range(num_labels):
        emotion_label = label_data[i, 0]-1  # python is 0-based
        print '\nProcessing Label %d' % (emotion_label+1)
        clip_bounds = label_data[i, :]

        for j in range(num_clips):
            print 'Clip %d' % (j+1)
            clip_id = num_clips*i + j
            clip_start = clip_bounds[2*j+1]
            clip_end = clip_bounds[2*j+2]
            print 'Clip bounds: ', clip_start, clip_end

            _, _, clip_length_original = calc_frame_boundaries(vs,
                                                               clip_start,
                                                               clip_end,
                                                               clip_scale_const[gender]) 
            print 'Original clip length: %d' % clip_length_original
 
            if include_expression_frames and emotion_label != 0:
                if j == 0:                    
                    clip_start -= clip_extend_amount[gender]                    
                    print 'New Clip bounds: ', clip_start, clip_end
                elif j == 2:
                    clip_end += clip_extend_amount[gender]                    
                    print 'New Clip bounds: ', clip_start, clip_end

            clip_frames, clip_timestamps = get_clip(vs, clip_start, clip_end, clip_scale_const[gender])
            num_frames_clip = clip_frames.shape[0]
            num_frames_total += num_frames_clip

            all_frames.append(clip_frames)
            subj_ids.append([subj_id]*num_frames_clip)
            clip_ids.append([clip_id]*num_frames_clip)
            emotion_labels.append([emotion_label]*num_frames_clip)
            timestamps.append(clip_timestamps)

            if include_expression_frames:
                easy_vector = numpy.zeros(num_frames_clip).astype('int32')
                if emotion_label != 0:
                    clip_extend_amount_frames = num_frames_clip - clip_length_original
                    print 'Number frames added: %d' % clip_extend_amount_frames

                    if j == 0:
                        easy_vector[0:clip_extend_amount_frames] = 1
                    elif j == 2:
                        easy_vector[-clip_extend_amount_frames:] = 1
                
                    print 'Num 0 frames: %d' % numpy.sum(easy_vector == 0) 
                    print 'Num 1 frames: %d' % numpy.sum(easy_vector == 1)

                easiness_labels.append(easy_vector)

    X = numpy.concatenate(all_frames, axis=0)

    subj_ids = numpy.hstack(subj_ids)
    clip_ids = numpy.hstack(clip_ids)
    timestamps = numpy.hstack(timestamps)
    emotion_labels = numpy.hstack(emotion_labels)

    if include_expression_frames:
        easiness_labels = numpy.hstack(easiness_labels)
        y = numpy.vstack([subj_ids, clip_ids, timestamps, easiness_labels, emotion_labels])
    else:
        y = numpy.vstack([subj_ids, clip_ids, timestamps, emotion_labels])

    return X, y


def save_to_npy_files(save_path, X, y, subj_id):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    numpy.save(os.path.join(save_path, 'X_'+subj_id+'.npy'), X)
    numpy.save(os.path.join(save_path, 'y_'+subj_id+'.npy'), y)


def parse_args():
    parser = argparse.ArgumentParser(description='Parse Chen_Huang dataset '
                                                 'for single subject and '
                                                 'extract images and labels.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_path', dest='video_path',
                        default='/data/Expr_Recog/Chen_Huang_avdata/ChenHuang_avi/',
                        help='Folder containing .avi files.')
    parser.add_argument('--label_path', dest='label_path',
                        default='/data/Expr_Recog/Chen_Huang_avdata/face/',
                        help='Folder containing .label files.')
    parser.add_argument('--subj_id', dest='subj_id',
                        help='Subject to extract frames and labels.')
    parser.add_argument('--save_path', dest='save_path',
                        default='/data/Expr_Recog/Chen_Huang_avdata_python/npy_files_raw/',
                        help='Folder to save output .npy files.')
    parser.add_argument('--include_expression_frames', dest='include_expression_frames',
                        action='store_true', default=False,
                        help='Flag to include frames with facial expression '
                             'but no audio.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Load command line arguments
    args = parse_args()
    video_path = args.video_path
    label_path = args.label_path
    subj_id = args.subj_id
    save_path = args.save_path
    include_expression_frames = args.include_expression_frames

    print '\nCommand Line Arguments: '
    print 'Video Path: %s' % video_path
    print 'Label Path: %s' % label_path
    print 'Save Path: %s' % save_path
    print 'Subject ID: %s' % subj_id

    all_subjs = ['03', '26', '46', '61', '68',
                 '81', '82', '85', '87', '90',
                 '04', '07', '16', '18', '19',
                 '31', '37', '56', '78', '94']
    # all_subjs = ['04', '07', '16', '18', '19', '31', '37', '56', '78', '94']
    # all_subjs = ['03', '26', '46', '61', '68', '81', '82', '85', '87', '90']
    all_genders = numpy.concatenate([['M']*10, ['F']*10])        

    # If no subject is specified, extract and save data for all 20 subjects
    if subj_id is None:
        print 'No subject specified.'
        print 'Extracting frames and labels for all 20 subjects.'
        for i, subj_id in enumerate(all_subjs):
            X, y = load_data_single_subj(subj_id, video_path, label_path,
                                         all_genders[i], include_expression_frames)
            print X.shape, y.shape

            print '\nSaving to .npy files.'
            save_to_npy_files(save_path, X, y, subj_id)

    else:
        # Check that subj_id is in all_subj
        try:
            ind = all_subjs.index(subj_id)
        except:
            print 'Subj_id %s does not exist in dataset.'
            sys.exit(0)

        gender = all_genders[ind]

        X, y = load_data_single_subj(subj_id, video_path, label_path,
                                     gender, include_expression_frames)
        print X.shape, y.shape

        print '\nSaving to .npy files.'
        save_to_npy_files(save_path, X, y, subj_id)

    print 'Done!'
