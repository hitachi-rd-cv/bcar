import os
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2

from torch.utils import data


class PRIDDataset(data.Dataset):

    img_directory = 'images'
    list_directory = 'lists'
    probe_list_footer = 'probe'
    gallery_list_footer = 'gallery'

    def __init__(self, root_path, targets):

        # When the class is used for trainig, "targets" must be list,
        # otherwise "targets" must be str
        assert type(targets) == list or type(targets) == str

        if type(targets) is str:
            self.name = targets

            probe_list_files = [list_file for list_file in os.listdir(
                os.path.join(root_path, self.list_directory)
            ) if targets in list_file and self.probe_list_footer in list_file]
            gallery_list_files = [list_file for list_file in os.listdir(
                os.path.join(root_path, self.list_directory)
            ) if targets in list_file and self.gallery_list_footer in list_file]

            assert len(probe_list_files) == len(gallery_list_files)

            probe_list_files.sort()
            gallery_list_files.sort()

            probe_files_for_all_sets = []
            gallery_files_for_all_sets = []
            for probe_list_file, gallery_list_file in zip(probe_list_files, gallery_list_files):
                with open(os.path.join(root_path, self.list_directory, probe_list_file)) as f:
                    probe_files_for_all_sets.append([line.rstrip('\n') for line in f.readlines()])
                with open(os.path.join(root_path, self.list_directory, gallery_list_file)) as f:
                    gallery_files_for_all_sets.append([line.rstrip('\n') for line in f.readlines()])

            self.probe_indices_for_all_sets = [[] for _ in range(len(probe_list_files))]
            self.gallery_indices_for_all_sets = [[] for _ in range(len(gallery_list_files))]

            targets = [targets]

        self.img_files = []
        self.identity_labels = []
        self.camera_labels = []
        self.class_n_imgs = []
        class_map = {}
        self.n_classes = 0
        for target in targets:
            img_files = os.listdir(os.path.join(root_path, self.img_directory, target))
            img_files.sort()

            for img_file in img_files:
                self.img_files.append(os.path.join(root_path, self.img_directory, target, img_file))

                idx = img_file.split('_')[1]
                class_name = target + '_' + idx
                if class_name not in class_map:
                    class_map[class_name] = self.n_classes
                    self.n_classes += 1
                    self.class_n_imgs.append(0)
                self.identity_labels.append(class_map[class_name])

                camera_id = int(img_file.split('_')[3])
                self.camera_labels.append(camera_id)

                self.class_n_imgs[class_map[class_name]] += 1

                if hasattr(self, 'probe_indices_for_all_sets'):
                    for probe_files, gallery_files, probe_indices, gallery_indices in zip(
                            probe_files_for_all_sets, gallery_files_for_all_sets,
                            self.probe_indices_for_all_sets, self.gallery_indices_for_all_sets
                    ):
                        assert not (img_file in probe_files and img_file in gallery_files)

                        if img_file in probe_files:
                            probe_indices.append(len(self.img_files)-1)
                        elif img_file in gallery_files:
                            gallery_indices.append(len(self.img_files)-1)

        self.class_n_imgs = np.array(self.class_n_imgs)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        identity_label = self.identity_labels[idx]
        camera_label = self.camera_labels[idx]

        return img, identity_label, camera_label

    def get_test_set(self):
        for probe_indices, gallery_indices in zip(
                self.probe_indices_for_all_sets, self.gallery_indices_for_all_sets
        ):
            yield probe_indices, gallery_indices

