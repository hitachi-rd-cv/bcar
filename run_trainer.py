import argparse

from lib import trainer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root_path', type=str, required=True,
    )
    parser.add_argument(
        '--log_path', type=str, required=True,
    )
    parser.add_argument(
        '--student_class_module', type=str, required=True,
    )
    parser.add_argument(
        '--teacher_class_module', type=str, required=True,
    )
    parser.add_argument(
        '--student_class_name', type=str, required=True,
    )
    parser.add_argument(
        '--teacher_class_name', type=str, required=True,
    )
    parser.add_argument(
        '--train_targets', type=str, nargs='+', required=True,
    )
    parser.add_argument(
        '--test_targets', type=str, nargs='+', required=True,
    )
    parser.add_argument(
        '--test_camera_base', action='store_true',
    )
    parser.add_argument(
        '--augmentation_types', type=str, nargs='+', default=[],
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
    )
    parser.add_argument(
        '--n_workers', type=int, default=7,
    )
    parser.add_argument(
        '--save_interval', type=int, default=1,
    )
    parser.add_argument(
        '--n_saved', type=int, default=1,
    )
    parser.add_argument(
        '--gpu_ids', type=int, nargs='+', default=[0],
    )
    parser.add_argument(
        '--max_epochs', type=int, default=150,
    )
    parser.add_argument(
        '--lr_decay_step', type=int, default=100,
    )
    parser.add_argument(
        '--init_lr_student_conv', type=float, default=.01,
    )
    parser.add_argument(
        '--init_lr_teacher_conv', type=float, default=.01,
    )
    parser.add_argument(
        '--init_lr_student_classifier', type=float, default=.01,
    )
    parser.add_argument(
        '--init_lr_teacher_classifier', type=float, default=.02,
    )
    parser.add_argument(
        '--init_interval', type=int, default=1,
    )
    parser.add_argument(
        '--hard_ratio', type=float, default=.3,
    )

    args = parser.parse_args()

    return args

def main(args):
    trainer.run(
        args.root_path, args.log_path, args.student_class_module,
        args.teacher_class_module, args.student_class_name, args.teacher_class_name, args.init_interval, args.hard_ratio,
        args.train_targets, args.test_targets, args.test_camera_base, args.augmentation_types, args.batch_size,
        args.n_workers, args.save_interval, args.n_saved, args.gpu_ids, max_epochs=args.max_epochs,
        init_lr_student_conv=args.init_lr_student_conv, init_lr_teacher_conv=args.init_lr_teacher_conv,
        init_lr_student_classifier=args.init_lr_student_classifier,
        init_lr_teacher_classifier=args.init_lr_teacher_classifier, lr_decay_step=args.lr_decay_step
    )

if __name__ == '__main__':
    args = get_args()
    main(args)
