import os
import csv
import json
import argparse
import os.path as osp
from tqdm import tqdm

from utils import *


def process_video(video_path, csv_path, dest_path, crop_size):

    splits = osp.splitext(csv_path)[0].split('_')
    start_frame_ind = int(splits[-2])
    end_frame_ind = int(splits[-1])

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if not osp.exists(dest_path):
        os.makedirs(dest_path)

    x_inds = []
    y_inds = []
    csv_reader = csv.reader(open(csv_path, 'r'))
    header = next(csv_reader)
    for i, item in enumerate(header):
        if item.startswith('featurePoint'):
            if item.endswith('X'):
                x_inds.append(i)
            elif item.endswith('Y'):
                y_inds.append(i)

    frame_ind = 0
    while True:
        ret, frame = cap.read()
        if ret:

            if frame_ind == end_frame_ind:
                cv2.destroyAllWindows()
                break

            if frame_ind >= start_frame_ind:
                csv_row = next(csv_reader)

                points = []
                for i in range(len(x_inds)):
                    x = int(round(float(csv_row[x_inds[i]]) * width))
                    y = int(round(float(csv_row[y_inds[i]]) * height))
                    points.append((x, y))

                aligned_face_crop = get_aligned_face_crop(frame, points, crop_size)

                frame_path = osp.join(dest_path, f'{frame_ind - start_frame_ind:04d}' + '.jpg')
                cv2.imwrite(frame_path, aligned_face_crop)

            frame_ind += 1

        else:
            break


def main(args):

    metadata = json.load(open(args.metadata_path, 'r'))
    delimiter = '_alt_seq_' if '_alt' in args.metadata_path else '_orig_seq_'
    for m in tqdm(metadata, ncols=100, desc='extracting face crops'):
        for tracking_data_path in metadata[m]['tracking_data']:
            video_path = tracking_data_path.split(delimiter)[0] + '.MP4'
            full_video_path = osp.join(args.raw_videos_root, video_path)
            full_csv_path = osp.join(args.track_data_root, tracking_data_path)
            full_dest_path = osp.join(args.cropped_faces_root, m, osp.splitext(osp.basename(video_path))[0])

            process_video(
                video_path=full_video_path,
                csv_path=full_csv_path,
                dest_path=full_dest_path,
                crop_size=args.crop_size
            )

    print('Face crop extraction finished!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', default='benchmark_data/CCMiniVID/CCMiniVID-O_metadata.json', type=str)
    parser.add_argument('--track_data_root', default='benchmark_data/CCMiniVID/tracking_data', type=str)
    parser.add_argument('--raw_videos_root', default='benchmark_data/CCMiniVID/raw_videos', type=str)
    parser.add_argument('--cropped_faces_root', default='benchmark_data/CCMiniVID/crops/CCMiniVID-O', type=str)
    parser.add_argument('--crop_size', default=256, type=int)
    args = parser.parse_args()

    main(args)
