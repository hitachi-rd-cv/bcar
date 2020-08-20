import numpy as np
import random
import cv2
import os
import shutil
from collections import defaultdict
import argparse


base_path = 'Persons'


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

    img_files = [f for f in os.listdir(os.path.join(args.root_path, base_path)) if f.split('.')[-1] == 'jpg']
    img_files.sort()
    id_camera_counts = defaultdict(lambda: 0)
    id_counts = defaultdict(lambda: 0)
    saved_files = []
    for img_file in img_files:
        person_id = int(img_file[:4])
        camera_id = 1
        key = '{}_{}'.format(person_id, camera_id)

        shutil.copy(os.path.join(args.root_path, base_path, img_file),
                    os.path.join(args.img_save_path, 'id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(
                        person_id, camera_id, id_camera_counts[key])))
        saved_files.append('id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(person_id, camera_id, id_camera_counts[key]))

        id_camera_counts[key] += 1
        id_counts[person_id] += 1

    counts = np.array(list(id_counts.values()))
    print('i-LIDS ID count statistics')
    print('Max: {}, Min: {}, Avg: {:.3f}'.format(np.amax(counts), np.amin(counts), np.mean(counts)))

    base_name = os.path.basename(os.path.normpath(args.img_save_path))
    person_ids = list(range(1, 120))
    for split in range(args.n_splits):
        random.shuffle(saved_files)
        random.shuffle(person_ids)
        probe_ids = []
        gallery_ids = []
        probe_files = []
        gallery_files = []
        for saved_file in saved_files:
            person_id = int(saved_file.split('_')[1])
            if person_id in person_ids[:60]:
                if person_id not in probe_ids:
                    probe_ids.append(person_id)
                    probe_files.append(saved_file)
                elif person_id not in gallery_ids:
                    gallery_ids.append(person_id)
                    gallery_files.append(saved_file)
        probe_files.sort()
        gallery_files.sort()
        with open(os.path.join(args.list_save_path, '{}_{}_probe.txt'.format(base_name, split)), 'w') as f:
            f.write('\n'.join(probe_files))
        with open(os.path.join(args.list_save_path, '{}_{}_gallery.txt'.format(base_name, split)), 'w') as f:
            f.write('\n'.join(gallery_files))


if __name__ == '__main__':
    args = get_args()
    main(args)
