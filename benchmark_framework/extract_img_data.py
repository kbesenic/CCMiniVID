import os
import dlib
import json
import argparse
import os.path as osp
from tqdm import tqdm


def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def extract_face_crop(frame_path, crop_path, points):
    frame = dlib.load_rgb_image(frame_path)

    if not osp.exists(osp.dirname(crop_path)):
        os.makedirs(osp.dirname(crop_path))

    mock_rectangle = dlib.rectangle(0, 0, 1, 1)  # bounding box is not required, alignment algo uses only points
    detection = dlib.full_object_detection(mock_rectangle, [dlib.point(p) for p in points])

    dlib.save_face_chip(frame, detection, chip_filename=crop_path, size=256, padding=0.5)


def main(args):
    metadata = load_metadata(args.detection_data_path)

    for frame_id in tqdm(metadata, ncols=100, desc='extracting face crops'):
        points = metadata[frame_id]
        frame_path = osp.join(args.raw_frames_root, frame_id[1:] + '.jpg')
        crop_path = osp.join(args.cropped_faces_root, frame_id[1:] + '.jpg')
        extract_face_crop(frame_path, crop_path, points)

    print('Face crop extraction finished!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_data_path', default='benchmark_data/CCMiniIMG/dlib_detections.json', type=str)
    parser.add_argument('--raw_frames_root', default='benchmark_data/CCMiniIMG/raw_frames', type=str)
    parser.add_argument('--cropped_faces_root', default='benchmark_data/CCMiniIMG/crops', type=str)
    args = parser.parse_args()

    main(args)
