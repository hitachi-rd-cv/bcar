import numpy as np
import os
import random
import shutil
from collections import defaultdict
import argparse


base_directory = 'single_shot'
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
    id_map = {}
    for camera in cameras:
        camera_id = 1 if camera == 'cam_a' else 2

        img_files = os.listdir(os.path.join(args.root_path, base_directory, camera))
        for img_file in img_files:
            person_id = int(img_file.rstrip('.png').split('_')[1])
            key = '{}_{}'.format(person_id, camera_id)

            shutil.copy(os.path.join(args.root_path, base_directory, camera, img_file),
                        os.path.join(args.img_save_path, 'id_{:05d}_cam_{:05d}_number_{:05d}.png'.format(
                            person_id, camera_id, id_camera_counts[key])))

            id_camera_counts[key] += 1
            id_counts[person_id] += 1

    counts = np.array(list(id_counts.values()))
    print('PRID ID count statistics')
    print('Max: {}, Min: {}, Avg: {:.3f}'.format(np.amax(counts), np.amin(counts), np.mean(counts)))

    base_name = os.path.basename(os.path.normpath(args.img_save_path))
    person_ids = list(range(1, 201))
    for split in range(args.n_splits):
        random.shuffle(person_ids)
        probe_files = ['id_{:05d}_cam_{:05d}_number_{:05d}.png'.format(person_id, 1, 0) for person_id in person_ids[:100]]
        gallery_files = ['id_{:05d}_cam_{:05d}_number_{:05d}.png'.format(person_id, 2, 0) for person_id in person_ids[:100]] + \
                        ['id_{:05d}_cam_{:05d}_number_{:05d}.png'.format(person_id, 2, 0) for person_id in range(201, 750)]
        probe_files.sort()
        gallery_files.sort()
        with open(os.path.join(args.list_save_path, '{}_{}_probe.txt'.format(base_name, split)), 'w') as f:
            f.write('\n'.join(probe_files))
        with open(os.path.join(args.list_save_path, '{}_{}_gallery.txt'.format(base_name, split)), 'w') as f:
            f.write('\n'.join(gallery_files))


if __name__ == '__main__':
    args = get_args()
    main(args)
