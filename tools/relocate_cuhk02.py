import numpy as np
import cv2
import os
import shutil
from collections import defaultdict
import argparse


directories = ('P1', 'P2', 'P3', 'P4', 'P5')
cameras = ('cam1', 'cam2')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path', type=str, required=True,
        help='Path to root directory'
    )
    parser.add_argument(
        '--save_path', type=str, required=True,
        help='Path to save directory'
    )

    return parser.parse_args()


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    id_map = defaultdict(lambda: max(id_map.values())+1)
    id_map['base'] = 0
    id_camera_counts = defaultdict(lambda: 0)
    id_counts = defaultdict(lambda: 0)
    camera_id = 1
    for directory in directories:
        for camera in cameras:
            img_files = [img_file for img_file in os.listdir(os.path.join(args.root_path, directory, camera))
                         if img_file.split('.')[-1] == 'png']
            img_files.sort()

            for img_file in img_files:
                person_id = id_map[directory+'_'+img_file[:3]]
                key = '{}_{}'.format(person_id, camera_id)

                shutil.copy(os.path.join(args.root_path, directory, camera, img_file),
                            os.path.join(args.save_path, 'id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(
                                person_id, camera_id, id_camera_counts[key])))

                id_camera_counts[key] += 1
                id_counts[person_id] += 1

            camera_id += 1

    counts = np.array(list(id_counts.values()))
    print('CUHK02 ID count statistics')
    print('Max: {}, Min: {}, Avg: {:.3f}'.format(np.amax(counts), np.amin(counts), np.mean(counts)))


if __name__ == '__main__':
    args = get_args()
    main(args)
