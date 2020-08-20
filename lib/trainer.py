import numpy as np
import os
import math
import importlib
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn, optim, utils
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, _prepare_batch
from ignite.engine.engine import Engine
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from lib.prid_dataset import PRIDDataset as Dataset
from lib.transformer import Transformer
from lib.rank_accuracy import RankAccuracy
from lib.soft_label_loss import SoftLabelLoss


class TrainDatasetWrapper(Dataset):

    def __init__(self, root_path, targets, transformer):
        super(TrainDatasetWrapper, self).__init__(root_path, targets)

        self.transformer = transformer

    def __getitem__(self, idx):
        img, identity_label, _ = super().__getitem__(idx)

        img = self.transformer.transform(img)
        identity_label = torch.tensor(identity_label)

        return img, identity_label


class TestDatasetWrapper(Dataset):

    def __init__(self, root_path, targets, transformer, return_camera=False):
        super(TestDatasetWrapper, self).__init__(root_path, targets)

        self.transformer = transformer
        self.return_camera = return_camera

    def __getitem__(self, idx):
        img, identity_label, camera_label = super().__getitem__(idx)

        img = self.transformer.transform(img)

        if self.return_camera:
            label = torch.tensor((identity_label, camera_label))
        else:
            label = torch.tensor(identity_label)

        return img, label


def create_supervised_soft_label_trainer(
        models, optimizers, loss_functions, hard_ratio, init_interval, device=None, non_blocking=False, prepare_batch=_prepare_batch,
        output_transform=lambda x, y, y_pred_student, y_pred_teacher, loss_student, loss_teacher: loss_student.item()):
    if device:
        for model in models.values():
            model.to(device)

    def _update(engine, batch):
        for name, model in models.items():
            if 'generator' in name:
                model.eval()
            else:
                model.train()

        for optimizer in optimizers.values():
            optimizer.zero_grad()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        with torch.no_grad():
            y_soft = F.softmax(models['generator'](x, return_logit=True), dim=1)

        y_pred_student = models['student'](x)
        loss_student = (1. - hard_ratio) * loss_functions['student'](y_pred_student, y_soft) + \
                       hard_ratio * loss_functions['teacher'](y_pred_student, y)

        y_pred_teacher = models['teacher'](x)
        loss_teacher = loss_functions['teacher'](y_pred_teacher, y)

        if engine.state.epoch > init_interval:
            loss_student.backward()
        loss_teacher.backward()

        for optimizer in optimizers.values():
            optimizer.step()

        return output_transform(x, y, y_pred_student, y_pred_teacher, loss_student, loss_teacher)

    return Engine(_update)


def _get_lr_decay_function(schedulers):

    def apply_lr_decay(engine):
        for scheduler in schedulers.values():
            scheduler.step()

    return apply_lr_decay


def _get_lr_write_function(optimizers, writer):

    def write_lr(engine):
        for name, optimizer in optimizers.items():
            param_group = optimizer.param_groups[0]
            lr = param_group['lr']
            writer.add_scalar('training/{}_lr'.format(name), lr, engine.state.epoch)

    return write_lr


def _get_loss_write_function(writer):

    def log_result(engine):
        writer.add_scalar('training/student_loss', engine.state.output[3].item(), engine.state.iteration)
        writer.add_scalar('training/teacher_loss', engine.state.output[4].item(), engine.state.iteration)

    return log_result


def _get_result_write_function(rank_accuracy, test_datasets, loader_caller, evaluator, writer):

    def log_result(engine):
        metrics = engine.state.metrics
        avg_loss_student = metrics['loss_student']
        avg_accuracy_student = metrics['accuracy_student']
        avg_loss_teacher = metrics['loss_teacher']
        avg_accuracy_teacher = metrics['accuracy_teacher']
        print('Training result - epoch: {}, average student loss: {:.4f}, average teacher loss: {:.4f}, '
              'average student accuracy: {:.4f}, average teacher accuracy: {:.4f}'
              .format(engine.state.epoch, avg_loss_student, avg_loss_teacher, avg_accuracy_student, avg_accuracy_teacher))
        writer.add_scalar('training/avg_loss_student', avg_loss_student, engine.state.epoch)
        writer.add_scalar('training/avg_accuracy_student', avg_accuracy_student, engine.state.epoch)
        writer.add_scalar('training/avg_loss_teacher', avg_loss_teacher, engine.state.epoch)
        writer.add_scalar('training/avg_accuracy_teacher', avg_accuracy_teacher, engine.state.epoch)

        for test_dataset in test_datasets:
            test_loader = loader_caller(test_dataset)
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            labels, features = metrics['rank']

            print(test_dataset.name)
            rank1_results = []
            rank5_results = []
            rank10_results = []
            for set_index, (probe_indices, gallery_indices) in enumerate(test_dataset.get_test_set()):
                rank_results, mean_ap = rank_accuracy.evaluate(
                    labels[probe_indices], labels[gallery_indices],
                    features[probe_indices], features[gallery_indices]
                )

                rank1_results.append(rank_results[0])
                rank5_results.append(rank_results[4])
                rank10_results.append(rank_results[9])

            rank1_avg = np.mean(rank1_results)
            rank5_avg = np.mean(rank5_results)
            rank10_avg = np.mean(rank10_results)

            print('Test Result - Epoch: {}, Average, Rank-1 accuracy: {:.4f}'
                  .format(engine.state.epoch, rank1_avg))
            print('Test Result - Epoch: {}, Average, Rank-5 accuracy: {:.4f}'
                  .format(engine.state.epoch, rank5_avg))
            print('Test Result - Epoch: {}, Average, Rank-10 accuracy: {:.4f}'
                  .format(engine.state.epoch, rank10_avg))
            print('Test Result - Epoch: {}, Mean AP: {:.4f}'
                  .format(engine.state.epoch, mean_ap))
            writer.add_scalar(
                'test/{}/avg/rank1'.format(test_dataset.name), rank1_avg, engine.state.epoch
            )
            writer.add_scalar(
                'test/{}/avg/rank5'.format(test_dataset.name), rank5_avg, engine.state.epoch
            )
            writer.add_scalar(
                'test/{}/avg/rank10'.format(test_dataset.name), rank10_avg, engine.state.epoch
            )
            writer.add_scalar(
                'test/{}/mean_ap'.format(test_dataset.name), mean_ap, engine.state.epoch
            )

    return log_result


def _get_test_data_loader_caller(batch_size, n_workers):

    def caller(dataset):
        return utils.data.DataLoader(dataset, batch_size, num_workers=n_workers, pin_memory=True)

    return caller


def _get_init_classifier_function(models, optimizer, init_interval):

    def init(engine):
        if engine.state.epoch % init_interval == 0:
            models['generator'].load_state_dict(models['teacher'].state_dict())

            optimizer.state[models['teacher'].classifier.weight]['momentum_buffer'][:] = 0.
            if models['teacher'].classifier.bias is not None:
                optimizer.state[models['teacher'].classifier.bias]['momentum_buffer'][:] = 0.

            models['teacher'].init_classifier()

    return init


def run(
        root_path, log_path, student_class_module, teacher_class_module, student_class_name,
        teacher_class_name, init_interval, hard_ratio, train_targets, test_targets, test_camera_base, augmentation_types,
        batch_size, n_workers, save_interval, n_saved, gpu_ids, max_epochs=150,
        init_lr_student_conv=.01, init_lr_teacher_conv=.01, init_lr_student_classifier=.01, init_lr_teacher_classifier=.1,
        lr_decay_step=100, lr_decay_rate=.1
):
    device = 'cuda:{}'.format(gpu_ids[0])

    train_transformer = Transformer(True, augmentation_types)
    test_transformer = Transformer(False, [])

    train_dataset = TrainDatasetWrapper(root_path, train_targets, train_transformer)
    train_loader = utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

    test_datasets = []
    for test_target in test_targets:
        test_datasets.append(TestDatasetWrapper(root_path, test_target, test_transformer, test_camera_base))

    loader_caller = _get_test_data_loader_caller(batch_size, n_workers)

    student_class_module = importlib.import_module(student_class_module)
    student_model_class = getattr(student_class_module, student_class_name)
    teacher_class_module = importlib.import_module(teacher_class_module)
    teacher_model_class = getattr(teacher_class_module, teacher_class_name)
    models = {
        'student': student_model_class(train_dataset.n_classes),
        'teacher': teacher_model_class(train_dataset.n_classes),
        'generator': teacher_model_class(train_dataset.n_classes)
    }

    loss_functions = {
        'student': SoftLabelLoss(),
        'teacher': nn.CrossEntropyLoss()
    }

    student_classifier_parameters = list(models['student'].classifier.parameters())
    student_classifier_parameters_ids = []
    for p in student_classifier_parameters:
        student_classifier_parameters_ids.append(id(p))
    student_conv_parameters = []
    for p in models['student'].parameters():
        if id(p) not in student_classifier_parameters_ids:
            student_conv_parameters.append(p)
    teacher_classifier_parameters = list(models['teacher'].classifier.parameters())
    teacher_classifier_parameters_ids = []
    for p in teacher_classifier_parameters:
        teacher_classifier_parameters_ids.append(id(p))
    teacher_conv_parameters = []
    for p in models['teacher'].parameters():
        if id(p) not in teacher_classifier_parameters_ids:
            teacher_conv_parameters.append(p)

    optimizers = {
        'student_conv': optim.SGD(student_conv_parameters, init_lr_student_conv, momentum=.9, weight_decay=5e-4, nesterov=True),
        'student_classifier': optim.SGD(student_classifier_parameters, init_lr_student_classifier, momentum=.9, weight_decay=5e-4,
                                        nesterov=True),
        'teacher_conv': optim.SGD(teacher_conv_parameters, init_lr_teacher_conv, momentum=.9, weight_decay=5e-4, nesterov=True),
        'teacher_classifier': optim.SGD(teacher_classifier_parameters, init_lr_teacher_classifier, momentum=.9, weight_decay=5e-4,
                                        nesterov=True),
    }

    schedulers = {
        'student_conv': optim.lr_scheduler.StepLR(optimizers['student_conv'], lr_decay_step, gamma=lr_decay_rate),
        'student_classifier': optim.lr_scheduler.StepLR(optimizers['student_classifier'], lr_decay_step, gamma=lr_decay_rate),
        'teacher_conv': optim.lr_scheduler.StepLR(optimizers['teacher_conv'], lr_decay_step, gamma=lr_decay_rate),
    }

    writer = SummaryWriter(log_dir=log_path)

    trainer = create_supervised_soft_label_trainer(
        models, optimizers, loss_functions, hard_ratio, init_interval, device=device, non_blocking=True,
        output_transform=lambda x, y, y_pred_student, y_pred_teacher, loss_student, loss_teacher: (
            y, y_pred_student, y_pred_teacher, loss_student, loss_teacher)
    )

    RunningAverage(output_transform=lambda output: output[3].item()).attach(trainer, 'loss_student')
    RunningAverage(output_transform=lambda output: output[4].item()).attach(trainer, 'loss_teacher')
    Accuracy(output_transform=lambda output: (output[1], output[0])).attach(trainer, 'accuracy_student')
    Accuracy(output_transform=lambda output: (output[2], output[0])).attach(trainer, 'accuracy_teacher')

    progress_bar = ProgressBar()
    progress_bar.attach(trainer, ['loss_student', 'loss_teacher'])

    checkpointer = ModelCheckpoint(
        log_path, 'checkpoint', save_interval=save_interval, n_saved=n_saved
    )

    rank_accuracy = RankAccuracy(n_workers)
    evaluator = create_supervised_evaluator(models['student'], metrics={'rank': rank_accuracy}, device=device, non_blocking=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, _get_result_write_function(
        rank_accuracy, test_datasets, loader_caller, evaluator, writer))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, _get_init_classifier_function(
        models, optimizers['teacher_classifier'], init_interval
    ))

    trainer.add_event_handler(Events.ITERATION_COMPLETED, _get_loss_write_function(writer))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, _get_lr_decay_function(schedulers))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, _get_lr_write_function(optimizers, writer))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'student_model': models['student'],
                                                                     'teacher_model': models['teacher'],
                                                                     'generator_model': models['generator']})

    trainer.run(train_loader, max_epochs=max_epochs)

    writer.close()
