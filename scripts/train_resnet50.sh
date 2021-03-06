python run_trainer.py \
       --root_path data \
       --log_path logs_resnet50 \
       --student_class_module lib.resnet \
       --student_class_name resnet50_ifn \
       --teacher_class_module lib.resnet \
       --teacher_class_name resnet50_ifn \
       --train_targets cuhk02 cuhk03 duke market person_search \
       --test_targets viper prid grid i_lids \
       --augmentation_types flip crop \
       --batch_size 64 \
       --n_workers 7 \
       --gpu_ids 0 \
       --init_lr_student_conv .005 \
       --init_lr_teacher_conv .005 \
       --init_lr_student_classifier .05 \
       --init_lr_teacher_classifier .02 \
       --init_interval 1 \
       --hard_ratio .3 \
       --max_epochs 70 \
       --lr_decay_step 40
