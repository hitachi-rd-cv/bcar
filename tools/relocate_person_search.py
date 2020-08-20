import numpy as np
import cv2
import os
from collections import defaultdict
import scipy.io
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

    mat = scipy.io.loadmat(os.path.join(args.root_path, 'annotation', 'Person.mat'))

    camera_id = 1

    id_camera_counts = defaultdict(lambda: 0)
    id_counts = defaultdict(lambda: 0)
    for i, person_info in enumerate(mat['Person'][0]):
        person_id = i + 1
        key = '{}_{}'.format(person_id, camera_id)

        for box_info in person_info[2][0]:
            img = cv2.imread(os.path.join(args.root_path, 'Image', 'SSM', box_info[0][0]))
            box = box_info[1][0].astype(np.int32)

            box[2:] += box[:2]
            img = img[box[1]:box[3], box[0]:box[2]]
            cv2.imwrite(os.path.join(args.save_path, 'id_{:05d}_cam_{:05d}_number_{:05d}.png'.format(
                person_id, camera_id, id_camera_counts[key])), img)

            id_camera_counts[key] += 1
            id_counts[person_id] += 1

    counts = np.array(list(id_counts.values()))
    print('PersonSearch ID count statistics')
    print('Max: {}, Min: {}, Avg: {:.3f}'.format(np.amax(counts), np.amin(counts), np.mean(counts)))

if __name__ == '__main__':
    args = get_args()
    main(args)
