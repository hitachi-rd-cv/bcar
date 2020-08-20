import numpy as np
import cv2
import os
import shutil
from collections import defaultdict
import argparse


cameras = ['probe', 'gallery']


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
    np.random.seed(args.random_seed)

    if not os.path.exists(args.img_save_path):
        os.makedirs(args.img_save_path)

    id_camera_counts = defaultdict(lambda: 0)
    id_counts = defaultdict(lambda: 0)
    probe_files = []
    true_gallery_files = []
    distractor_gallery_files = []
    for camera in cameras:
        camera_id = cameras.index(camera) + 1

        img_files = [f for f in os.listdir(os.path.join(args.root_path, camera)) if f.split('.')[-1] == 'jpeg']

        for img_file in img_files:
            person_id = int(img_file.split('_')[0])

            key = '{}_{}'.format(person_id, camera_id)

            if camera == 'probe':
                probe_files.append('id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(person_id, camera_id, id_camera_counts[key]))
            else:
                if person_id == 0:
                    distractor_gallery_files.append('id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(person_id, camera_id,
                                                                                                    id_camera_counts[key]))
                else:
                    true_gallery_files.append('id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(person_id, camera_id,
                                                                                              id_camera_counts[key]))

            shutil.copy(os.path.join(args.root_path, camera, img_file),
                        os.path.join(args.img_save_path, 'id_{:05d}_cam_{:05d}_number_{:05d}.jpg'.format(
                            person_id, camera_id, id_camera_counts[key])))

            id_camera_counts[key] += 1
            id_counts[person_id] += 1

    counts = np.array(list(id_counts.values()))
    print('GRID ID count statistics')
    print('Max: {}, Min: {}, Avg: {:.3f}'.format(np.amax(counts), np.amin(counts), np.mean(counts)))

    probe_files.sort()
    true_gallery_files.sort()
    distractor_gallery_files.sort()

    base_name = os.path.basename(os.path.normpath(args.img_save_path))
    probe_files = np.array(probe_files)
    true_gallery_files = np.array(true_gallery_files)
    indices = np.arange(probe_files.shape[0])
    for split in range(args.n_splits):
        np.random.shuffle(indices)
        selected_indices = indices[:125]
        selected_indices = np.sort(selected_indices)
        with open(os.path.join(args.list_save_path, '{}_{}_probe.txt'.format(base_name, split)), 'w') as f:
            f.write('\n'.join(probe_files[selected_indices].tolist()))
        with open(os.path.join(args.list_save_path, '{}_{}_gallery.txt'.format(base_name, split)), 'w') as f:
            f.write('\n'.join(true_gallery_files[selected_indices].tolist()+distractor_gallery_files))


if __name__ == '__main__':
    args = get_args()
    main(args)
