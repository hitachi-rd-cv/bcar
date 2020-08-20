import numpy as np
import cv2
import os
import shutil
import random
from collections import defaultdict
import argparse


cameras = ['cam_a', 'cam_b']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path', type=str, required=True,
        help='Path to root directory'
    )
    parser.add_argument(
        '--img_save_path', type=str, required=True,
        help='Path to save directory'
    )
    parser.add_argument(
        '--list_save_path', type=str, required=True,
        help='Path to save directory'
    )
    parser.add_argument(
        '--n_splits', type=int, default=10,
        help='The number of splits'
    )
    parser.add_argument(
        '--random_seed', type=int, default=0,
        help='Random seed'
    )

    return parser.parse_args()


def main(args):
    random.seed(args.random_seed)

    if not os.path.exists(args.img_save_path):
        os.makedirs(args.img_save_path)

    id_camera_counts = defaultdict(lambda: 0)
    id_counts = defaultdict(lambda: 0)
    for camera in cameras:
        camera_id = cameras.index(camera) + 1

        img_files = [f for f in os.listdir(os.path.join(args.root_path, camera)) if f.split('.')[-1] == 'bmp']

        for img_file in img_files:
            person_id = int(img_file.split('_')[0])
            key = '{}_{}'.format(person_id, camera_id)

            shutil.copy(os.path.join(args.root_path, camera, img_file),
                os.path.join(args.img_save_path, 'id_{:05d}_cam_{:05d}_number_{:05d}.bmp'.format(
                    person_id, camera_id, id_camera_counts[key])))

            id_camera_counts[key] += 1
            id_counts[person_id] += 1

    counts = np.array(list(id_counts.values()))
    print('VIPeR ID count statistics')
    print('Max: {}, Min: {}, Avg: {:.3f}'.format(np.amax(counts), np.amin(counts), np.mean(counts)))

    base_name = os.path.basename(os.path.normpath(args.img_save_path))
    person_ids = list(id_counts.keys())
    for split in range(args.n_splits):
        random.shuffle(person_ids)
        probe_files = ['id_{:05d}_cam_{:05d}_number_{:05d}.bmp'.format(person_id, 1, 0) for person_id in
                       person_ids[:len(person_ids)//2]]
        gallery_files = ['id_{:05d}_cam_{:05d}_number_{:05d}.bmp'.format(person_id, 2, 0) for person_id in
                         person_ids[:len(person_ids)//2]]
        probe_files.sort()
        gallery_files.sort()
        with open(os.path.join(args.list_save_path, '{}_{}_probe.txt'.format(base_name, split)), 'w') as f:
            f.write('\n'.join(probe_files))
        with open(os.path.join(args.list_save_path, '{}_{}_gallery.txt'.format(base_name, split)), 'w') as f:
            f.write('\n'.join(gallery_files))


if __name__ == '__main__':
    args = get_args()
    main(args)
