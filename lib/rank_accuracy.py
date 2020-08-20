import numpy as np
import multiprocessing as mp

from ignite.metrics.metric import Metric


class RankAccuracy(Metric):

    def __init__(self, n_workers):
        super(RankAccuracy, self).__init__()

        self.n_workers = n_workers

    def reset(self):
        self.labels = []
        self.features = []
        self.miss_probe_labels = None
        self.miss_gallery_labels = None

    def _euclidian_distance(self, features1, features2):
        d = np.sum(
            features1*features1, axis=1, keepdims=True
        ) + np.sum(
            features2*features2, axis=1, keepdims=True
        ).T - 2. * np.matmul(features1, features2.T)

        return d

    def update(self, output):
        features, labels = output

        self.labels.append(labels.detach().to('cpu').numpy())
        self.features.append(features.detach().to('cpu').detach().numpy())

    def compute(self):
        labels = np.concatenate(self.labels, axis=0)
        features = np.concatenate(self.features, axis=0)

        return labels, features

    def evaluate(self, probe_labels, gallery_labels, probe_features, gallery_features):
        n_queries = probe_labels.shape[0]
        n_galleries = gallery_labels.shape[0]

        distances = self._euclidian_distance(probe_features, gallery_features)

        pool = mp.Pool(self.n_workers)

        results = [pool.apply_async(
            compute_ap, args=(probe_labels[i], gallery_labels, distances[i, :])
        ) for i in range(n_queries)]

        pool.close()
        pool.join()

        ap = np.zeros(n_queries, dtype='float32')
        cmc = np.zeros((n_queries, n_galleries), dtype='float32')
        for i in range(n_queries):
            ap[i], cmc[i] = results[i].get()

        cmc = cmc[ap>=0.]
        ap = ap[ap>=0.]

        rank_accuracy = np.mean(cmc, axis=0)
        mean_ap = np.mean(ap)

        if probe_labels.ndim == 2:
            self.miss_probe_labels = probe_labels[cmc[:, 0]==0, 0]
            self.miss_gallery_labels = gallery_labels[np.argmin(distances[cmc[:, 0]==0], axis=1), 0]
        else:
            self.miss_probe_labels = probe_labels[cmc[:, 0]==0]
            self.miss_gallery_labels = gallery_labels[np.argmin(distances[cmc[:, 0]==0], axis=1)]

        return rank_accuracy, mean_ap


# compute_ap have to be a global method because an instance method cannot be used for multiprocessing
def compute_ap(probe_label, gallery_labels, distances):
    if probe_label.size == 2:
        probe_id = probe_label[0]
        probe_camera = probe_label[1]

        gallery_ids = gallery_labels[gallery_labels[:, 1]!=probe_camera, 0]
        distances = distances[gallery_labels[:, 1]!=probe_camera]
    else:
        probe_id = probe_label
        gallery_ids = gallery_labels

    indices = np.argsort(distances)
    correct_indices = np.where(gallery_ids==probe_id)[0]

    cmc = np.zeros(gallery_labels.shape[0], dtype='int32')
    n_corrects = correct_indices.size

    if n_corrects == 0:
        return -1, cmc

    old_recall = 0.
    old_precision = 1.0
    ap = 0.
    n_current_corrects = 0
    for i in range(indices.shape[0]):
        if (correct_indices == indices[i]).any():
            cmc[i:] = 1
            n_current_corrects += 1

        recall = n_current_corrects / n_corrects
        precision = n_current_corrects / (i + 1)
        ap += (recall - old_recall) * ((old_precision + precision) / 2.)
        old_recall = recall
        old_precision = precision

        if n_current_corrects == n_corrects:
            return ap, cmc
