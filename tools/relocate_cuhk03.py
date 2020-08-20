import numpy as np
import h5py
import cv2
import os
from collections import defaultdict
import argparse


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

    mat = h5py.File(os.path.join(args.root_path, 'cuhk-03.mat'), 'r')
    dataset = mat['detected']
    id_camera_counts = defaultdict(lambda: 0)
    id_counts = defaultdict(lambda: 0)
    person_id_offset = 0
    for camera_pair_id in range(5):
        camera_id1 = camera_pair_id * 2 + 1
        camera_id2 = camera_pair_id * 2 + 2
        camera_pair_data = mat[dataset[0, camera_pair_id]]
        for img_index in range(camera_pair_data.shape[0]):
            for person_id in range(camera_pair_data.shape[1]):
                img = mat[camera_pair_data[img_index, person_id]][:3]

                if img.shape[0] != 3:
                    continue

                img = img[::-1]
                img = img.T

                if img_index < 5:
                    camera_id = camera_id1
                else:
                    camera_id = camera_id2

                key = '{}_{}'.format(person_id+person_id_offset, camera_id)

                cv2.imwrite(os.path.join(args.save_path, 'id_{:05d}_cam_{:05d}_number_{:05d}.png'.format(
                    person_id+person_id_offset, camera_id, id_camera_counts[key])), img)

                id_camera_counts[key] += 1
                id_counts[person_id+person_id_offset] += 1

        person_id_offset += camera_pair_data.shape[1]

    counts = np.array(list(id_counts.values()))
    print('CUHK03 ID count statistics')
    print('Max: {}, Min: {}, Avg: {:.3f}'.format(np.amax(counts), np.amin(counts), np.mean(counts)))


if __name__ == '__main__':
    args = get_args()
    main(args)
