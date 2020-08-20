import numpy as np
import cv2
import os
from collections import defaultdict
import argparse
import shutil


directories = ['bounding_box_train', 'bounding_box_test', 'query']


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
        '--original', action='store_true',
        help='Whether to save original train and test set or not'
    )

    return parser.parse_args()


def main(args):
    if not os.path.exists(args.img_save_path):
        os.makedirs(args.img_save_path)
    if args.original:
        if not os.path.exists('{}_train'.format(args.img_save_path)):
            os.makedirs('{}_train'.format(args.img_save_path))
        if not os.path.exists('{}_test'.format(args.img_save_path)):
            os.makedirs('{}_test'.format(args.img_save_path))

    id_camera_counts = defaultdict(lambda: 0)
    id_counts = defaultdict(lambda: 0)
    probe_files = []
    gallery_files = []
    for directory in directories:
        img_files = [f for f in os.listdir(os.path.join(args.root_path, directory)) if f.split('.')[-1] == 'jpg']

        for img_file in img_files:
            elements = img_file.split('_')

            person_id = int(elements[0])

            if person_id == -1:
                continue

            camera_id = int(elements[1][1])
            key = '{}_{}'.format(person_id, camera_id)

            if  person_id != 0:
                shutil.copy(os.path.join(args.root_path, directory, img_file),
                            os.path.join(args.img_save_path, 'id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(
                                person_id, camera_id, id_camera_counts[key])))

            if args.original:
                if 'train' in directory:
                    shutil.copy(os.path.join(args.root_path, directory, img_file),
                                os.path.join('{}_train'.format(args.img_save_path), 'id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(
                                    person_id, camera_id, id_camera_counts[key])))
                else:
                    shutil.copy(os.path.join(args.root_path, directory, img_file),
                                os.path.join('{}_test'.format(args.img_save_path), 'id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(
                                    person_id, camera_id, id_camera_counts[key])))

            if directory == 'query':
                probe_files.append('id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(person_id, camera_id, id_camera_counts[key]))
            elif directory == 'bounding_box_test':
                gallery_files.append('id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(person_id, camera_id, id_camera_counts[key]))

            id_camera_counts[key] += 1
            if person_id != 0:
                id_counts[person_id] += 1

    counts = np.array(list(id_counts.values()))
    print('Duke ID count statistics')
    print('Max: {}, Min: {}, Avg: {:.3f}'.format(np.amax(counts), np.amin(counts), np.mean(counts)))

    base_name = os.path.basename(os.path.normpath(args.img_save_path))
    probe_files.sort()
    gallery_files.sort()
    with open(os.path.join(args.list_save_path, '{}_test_probe.txt'.format(base_name)), 'w') as f:
        f.write('\n'.join(probe_files))
    with open(os.path.join(args.list_save_path, '{}_test_gallery.txt'.format(base_name)), 'w') as f:
        f.write('\n'.join(gallery_files))


if __name__ == '__main__':
    args = get_args()
    main(args)
