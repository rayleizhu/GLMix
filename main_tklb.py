"""
IN1k training script with token labeling. 
Based on https://github.com/YehLi/ImageNetModel/blob/main/classification/main.py

Modifications:
* copy https://github.com/zihangJiang/TokenLabeling/tree/main/tlt/data locally to avoid extra package (tlt) installation
* logging, slurm etc.

"""
import argparse
import datetime
import os
import numpy as np
import time
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import json

from pathlib import Path
import timm
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, distribute_bn
from timm.models.layers import trunc_normal_

# TODO: clean up token_labeling dependencies
from token_labeling.engine import train_one_epoch, evaluate
from token_labeling.tlt_data import create_token_label_target, TokenLabelMixup, FastCollateTokenLabelMixup, \
    create_token_label_loader, create_token_label_dataset
from token_labeling.loss import TokenLabelGTCrossEntropy, TokenLabelCrossEntropy, TokenLabelSoftTargetCrossEntropy

from fvcore.nn import FlopCountAnalysis, flop_count_table

import hydra
from omegaconf import DictConfig, OmegaConf

import ConvNextV2.utils as cx2utils
from benchmark import InferenceBenchmarkRunner
import models.glnet_tklb

# import warnings
# warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def build_token_label(args):
    dataset_train = create_token_label_dataset(
            '', root=args.data_path, label_root=args.token_label_data,
            use_memcache=args.use_memcache,
            mc_config_path=args.mc_config_path) if not args.eval else None
    dataset_eval = timm.data.create_dataset('',
            root=args.data_path,
            split='validation',
            is_training=False,
            batch_size=int(1.5 * args.batch_size))

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    args.mixup = 0.0
    args.cutmix = 0.0
    args.cutmix_minmax = None
    args.train_interpolation = 'random'
    args.prefetcher = True
    num_aug_splits = 0
    args.pin_mem = False

    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.token_label and args.token_label_data:
        use_token_label = True
    else:
        use_token_label = False
    
    loader_train = create_token_label_loader(
        dataset_train,
        input_size=(3, args.input_size, args.input_size),
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=timm.data.constants.IMAGENET_DEFAULT_MEAN,
        std=timm.data.constants.IMAGENET_DEFAULT_STD,
        num_workers=args.num_workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        use_token_label=use_token_label) if not args.eval else None

    loader_eval = timm.data.create_loader(
        dataset_eval,
        input_size=(3, args.input_size, args.input_size),
        batch_size=int(1.5 * args.batch_size),
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation='bicubic',
        mean=timm.data.constants.IMAGENET_DEFAULT_MEAN,
        std=timm.data.constants.IMAGENET_DEFAULT_STD,
        num_workers=args.num_workers,
        distributed=args.distributed,
        crop_pct=0.96,
        pin_memory=args.pin_mem,
        persistent_workers=False
    )
    return dataset_train, loader_train, dataset_eval, loader_eval, mixup_fn, 1000

def build_imagenet_dataset(args):
    assert args.token_label_data
    return build_token_label(args)

def main(args):
    cx2utils.init_distributed_mode(args)
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + cx2utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    global_rank = cx2utils.get_rank()
    num_tasks = cx2utils.get_world_size()

    dataset_train, data_loader_train, dataset_val, data_loader_val, mixup_fn, args.nb_classes \
        = build_imagenet_dataset(args)

    #################### Throughput Benchmark ##################################
    bench_results = None
    if global_rank == 0 and args.benchmark:
        print('Run inference throughput benchmark')
        bench = InferenceBenchmarkRunner(
            model_name=args.model, device='cuda', batch_size=128)
        torch.cuda.empty_cache()
        bench_results = bench.run()
        print(f'--benchmark result\n{json.dumps(bench_results, indent=4)}')
        del bench
        torch.cuda.empty_cache()
    
    #################### Create Model ###########################################
    print(f"Creating model: {args.model}")
    model:nn.Module = create_model(
        args.model,
        pretrained=args.load_release,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        token_label=args.token_label,
    )

    # TODO: check if we need the line below
    # model.eval()

    #################### Flops Analysis Table ###################################
    if global_rank == 0:
        flops = FlopCountAnalysis(model, torch.rand(1, 3, args.input_size, args.input_size))
        print(flop_count_table(flops))

    #################### finetune if specified ###################################
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        cx2utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        
        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        torch.nn.init.constant_(model.head.bias, 0.)

    model.to(device)

    ####################### create model ema if specified #######################
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    #################### model specs & DDP wrapper #############################
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    ############################## lr scaling & optimizer & loss #########################
    linear_scaled_lr = args.lr * args.batch_size * cx2utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    print(f"Linearly scaled lr {linear_scaled_lr}!")

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = TokenLabelCrossEntropy(dense_weight=args.dense_weight, \
        cls_weight=args.cls_weight, mixup_active=False).cuda()

    ############################## auto resume #########################################
    # NOTE:auto_load_model may recover args._max_accuracy,  args._max_accuracy_ema, args._pavi_tid
    args._max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        args._max_accuracy_ema = 0.0
    args._pavi_tid = None # for pavi logger resume
    
    cx2utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    
    if args.distributed:
        dist.barrier()
    
    ############################# eval mode shortcut if specified ####################################
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    ############################# training visualizer ####################################
    # NOTE: logging is only needed for trainining, hence put logger initialization after eval shortcut
    if global_rank == 0 and hasattr(args, "pavi"):
        assert args.pavi.name and args.pavi.tag # required args
        if args._pavi_tid is not None:
            print(f'resume logging to pavi (training_id={args._pavi_tid})')
        # if log_dir is set, use it
        # else use args.pavi.log_dir for local storage of events
        log_dir = args.log_dir or args.pavi.log_dir
        log_writer = cx2utils.PaviLogger(
            # log_dir=args.pavi.log_dir,
            # NOTE: hack, see if the output is compatiable with tensorboard
            log_dir=log_dir,
            training_id=args._pavi_tid,
            name=args.pavi.name,
            project=args.pavi.project,
            tags=args.pavi.tag,
            description=args.pavi.description
        )
        args._pavi_tid = log_writer.training_id # to be checkpointed for resume
        print(f'Pavi logger is created (training_id={args._pavi_tid})!  ')
    elif global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = cx2utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if (log_writer is not None) and (bench_results is not None):
        log_writer.update(head='bench', step=0, **bench_results)

    ############################## training loop ###########################
    dist.barrier()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        if log_writer is not None:
            # NOTE: We use epoch_1000x as the x-axis in tensorboard. 
            # This calibrates different curves when batch size changes.
            log_writer.set_step(epoch*1000)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            fp32=args.fp32_resume, args=args,
            log_writer=log_writer
        )

        if args.distributed:
            print('Distributing BatchNorm running means and vars')
            distribute_bn(model, cx2utils.get_world_size(), True)

        lr_scheduler.step(epoch)

        ############# logging & checkpointing routines ##############
        if log_writer is not None:
            log_writer.update(head="epoch_all_proc_avg", step=epoch, **train_stats)
        
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                cx2utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        
        ######################### eval & save best ###################
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if args._max_accuracy < test_stats["acc1"]:
                args._max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    cx2utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {args._max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if args._max_accuracy_ema < test_stats_ema["acc1"]:
                    args._max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        cx2utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {args._max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and cx2utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if log_writer is not None:
        log_writer.close()



@hydra.main(version_base=None, config_path="./configs", config_name="main_tklb.yaml")
def hydra_app(args:DictConfig):
    # NOTE: enable write to unknow field of cfg
    # hence it behaves like argparse.NameSpace
    # https://stackoverflow.com/a/66296809
    OmegaConf.set_struct(args, False)

    if not hasattr(args, 'slurm'): # run locally
        if args.output_dir is not None:
            # https://stackoverflow.com/a/73007887, read config in "hydra" section
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            args.output_dir = hydra_cfg['runtime']['output_dir'] # path where to save
        main(args=args)
        
    else:
        from slurm_wrapper import run_with_submitit
        if args.slurm.job_dir is None:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            args.slurm.job_dir = hydra_cfg['runtime']['output_dir']
            args.output_dir = args.slurm.job_dir
        # NOTE: main cannot be pickled and cause an error, hence my solution is
        # import main to slurm wrapper according to args._entrance_id (a string)
        # see slurm_wrapper.get_main_func_entrance()
        args._entrance_id = 'main_tklb'
        run_with_submitit(args=args) 


if __name__ == '__main__':
    hydra_app()