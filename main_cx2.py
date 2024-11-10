"""
This script is modified from ConvNextV2's finetuning script:
https://github.com/facebookresearch/ConvNeXt-V2/blob/main/main_finetune.py

All modifications are for user experience only, including:

* replacing argparse with hydra for easier arguments management (add, change, delete)
* automatically generating experiment directory according to argument overrides
* adding submitit support for smooth experience on slurm cluster (see also slurm_wrapper.py)
* creating model with timm.create_model(), so users can manage models easily with timm.register_model()
* pavi (SenseTime intra-net training visualization) logger
* flops count table & inference throughput benchmark

No modifcations on training logic or hyperparameters.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DistributedSampler,SequentialSampler, DataLoader

from fvcore.nn import FlopCountAnalysis, flop_count_table

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from timm import create_model

import hydra
from omegaconf import DictConfig, OmegaConf

# TODO: gradually remove the dependency on submodule
from ConvNextV2.optim_factory import create_optimizer, LayerDecayValueAssigner
from ConvNextV2.datasets import build_dataset
from ConvNextV2.engine_finetune import train_one_epoch, evaluate
import ConvNextV2.utils as cx2utils
from ConvNextV2.utils import NativeScalerWithGradNormCount as NativeScaler

from benchmark import InferenceBenchmarkRunner
from samplers import RASampler
import models.glnet


def main(args):
    cx2utils.init_distributed_mode(args)
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + cx2utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = cx2utils.get_world_size()
    global_rank = cx2utils.get_rank()
    if args.repeated_aug:
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
        )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = SequentialSampler(dataset_val)

    _DL = DataLoader
    if args.use_prefetch:
        from prefetch_generator import DataLoaderX
        _DL = DataLoaderX 

    data_loader_train = _DL(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=args.persistent_workers, # maybe important to avoid memory leak of lmdb?
        drop_last=True,
    )
    if dataset_val is not None:
        data_loader_val = _DL(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            persistent_workers=args.persistent_workers,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
   
    # BUG: TypeError: __init__() got an unexpected keyword argument 'global_pool'
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

    print(f"Creating model: {args.model}")
    model:nn.Module = create_model(
        model_name=args.model,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        pretrained=args.load_release
    )

    # print(model)
    model.eval()

    if global_rank == 0:
        flops = FlopCountAnalysis(model, torch.rand(1, 3, args.input_size, args.input_size))
        print(flop_count_table(flops))

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

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    eff_batch_size = args.batch_size * args.update_freq * cx2utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size
    
    if args.lr is None:
        # NOTE: this is different from uniformer, where the divisor is 512 instead of 256
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    # TODO: Implement layer_decay later
    # layer_decay is only used for further finetuning with 384x384 resolution
    # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/main.py#L92
    # https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/main.py#L335
    # https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md#imagenet-1k-fine-tuning
    # if args.layer_decay < 1.0 or args.layer_decay > 1.0:
    #     assert args.layer_decay_type in ['single', 'group']
    #     if args.layer_decay_type == 'group': # applies for Base and Large models
    #         num_layers = 12
    #     else:
    #         num_layers = sum(model_without_ddp.depths)
    #     assigner = LayerDecayValueAssigner(
    #         list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
    #         depths=model_without_ddp.depths, layer_decay_type=args.layer_decay_type)
    # else:
    #     assigner = None
    assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.ddp_fup)
        model_without_ddp = model.module

    # TODO: support model.no_weight_decay()
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    print("criterion = %s" % str(criterion))

    
    # NOTE:auto_load_model may recover args._max_accuracy,  args._max_accuracy_ema, args._pavi_tid
    args._max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        args._max_accuracy_ema = 0.0
    args._pavi_tid = None # for pavi logger resume
    
    cx2utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the model on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        # if args.model_ema and args.model_ema_eval:
        #     test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
        #     print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
        return
    
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
    elif global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = cx2utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if (log_writer is not None) and (bench_results is not None):
        log_writer.update(head='bench', step=0, **bench_results)

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            # log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            # NOTE: We use epoch_1000x as the x-axis in tensorboard. 
            # This calibrates different curves when batch size changes.
            log_writer.set_step(epoch*1000)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler, 
            args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        if log_writer is not None:
            log_writer.update(head="epoch_all_proc_avg", step=epoch, **train_stats)

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                cx2utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
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


@hydra.main(version_base=None, config_path="./configs", config_name="main_cx2.yaml")
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
        args._entrance_id = 'main_cx2'
        run_with_submitit(args=args) 

if __name__ == '__main__':
    hydra_app()