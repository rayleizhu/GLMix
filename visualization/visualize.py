
import torch
from torch import nn
import torch.utils.data
from timm.models import create_model
from functools import partial
import math
# from torchvision.utils import draw_segmentation_masks
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
import os.path as osp
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np

import sys
project_root = osp.dirname(osp.dirname(os.path.abspath(__file__)))
print(f"append projection root {project_root} to PYTHONPATH.")
sys.path.append(project_root)
from datasets import build_dataset
from visualization.utils import visualize_a_batch
from models.glnet 
import models.glnet_tklb

from argparse import Namespace


def main(args:Namespace):
    ################ create a model and register the hook #############
    model:nn.Module = create_model(args.model, num_classes=1000, pretrained=args.load_release)
    device = torch.device(args.device)
    model = model.to(device)
    # print(model)

    # load checkpoint
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        msg = model.load_state_dict(ckpt['model'])
        print(msg)

    # register forward hook
    hooked_tensors = []
    for name, mod in model.named_modules():
        if hasattr(mod, "vis_proxy"):
            print(f"found visualization proxy in module {name}")
            assert isinstance(mod.vis_proxy, nn.Identity)
            mod.vis_proxy.register_forward_hook(
                lambda self, input, output: hooked_tensors.append(output))
    model.eval()

    ################ reproducible data loader #############
    dataset_val, _ = build_dataset(is_train=False, args=args)
    print(f"number of samples in dataset: {len(dataset_val)}.")
    # https://blog.csdn.net/qq_40475568/article/details/118959964
    # NOTE: this may be unnecessary, keep it for safety
    def worker_init_fn(worker_id, seed):
        random.seed(seed + worker_id)
    g = torch.Generator()
    g.manual_seed(args.seed)
    # sampler = torch.utils.data.SequentialSampler(dataset_val)
    sampler = torch.utils.data.RandomSampler(dataset_val, num_samples=args.num_samples, generator=g)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        worker_init_fn=partial(worker_init_fn, seed=args.seed),
        generator=g)

    ################ visualize ###################
    print(f"block ids for visualization {args.vis_block_ids}.")
    if not args.save_dir:
        ckpt_id = Path(args.ckpt).stem if args.ckpt else f'load_release_{args.load_release}'
        save_dir = osp.join(
            "outputs", "visualization", 
            ".".join([args.model, f"block{'_'.join([ str(x) for x in args.vis_block_ids ])}",
                ckpt_id, f"seed{args.seed}"])
        )
    else:
        save_dir = args.save_dir
    print(f"visualizations will be saved to {save_dir}.")
    os.makedirs(save_dir, exist_ok=True)

    cnt = 0
    for batch_id, (im_batch, label) in tqdm(enumerate(data_loader_val)):
        hooked_tensors = [] # clear hooked tensors
        im_batch = im_batch.to(device)
        _ = model.forward(im_batch)

        vis_assignments = []
        for block_id in args.vis_block_ids:
            assignment_logits, attn_weights = hooked_tensors[block_id]
            bs, n_clusters, n_feats = assignment_logits.size()
            res = math.isqrt(assignment_logits.size(-1))
            assignment = torch.softmax(assignment_logits, dim=1)
            assignment = assignment.view(bs, n_clusters, res, res).detach().cpu()
            vis_assignments.append(assignment)

        in1k_mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).view(1, 3, 1, 1)
        in1k_std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).view(1, 3, 1, 1)
        imgs = im_batch * in1k_std + in1k_mean
        imgs = (imgs*255).to(torch.uint8).cpu()

        # begin = 108
        # end = 128
        begin = 0
        end = 16
        resolution = 128
        gray_cmap = np.reshape(np.linspace(begin, end, resolution), (resolution, 1, 1))
        gray_cmap = np.tile(gray_cmap, (256//resolution, 1, 3)).astype(np.uint8)
        # cmap = cv2.COLORMAP_BONE
        cmap = gray_cmap

        vis = visualize_a_batch(imgs, vis_assignments, cmap=cmap,
            heat_intp='nearest', highlight_thickness=args.hl_thickness)
        
        for i, grid in enumerate(vis):
            save_path = osp.join(save_dir, f'{cnt+i:05d}.png')
            # print(save_path)
            grid.save(save_path)
        cnt += len(vis)


if __name__ == "__main__":
    args = Namespace(
        model = 'glnet_stl',
        load_release = True,
        ckpt = None, # local checkpoint, will always override the released on
        # ckpt = 'outputs/glnet_stl/cls/20231012-14.21.57/checkpoint-best.pth',
        # model = 'glnet_stl_paramslot',
        # ckpt = 'outputs_release/batch_size.128-drop_path.0.10-lr.2e-3-model.glnet_stl_paramslot-slurm.nodes.2/cls/20231012-14.21.46/checkpoint-best.pth',
        vis_block_ids = (4,), # you can set multiple blocks for visualizetion, e.g., (1,3,4,9)
        save_dir=None, # by setting save_dir None, the default output path will be like "outputs/visualization/glnet_stl.block4.checkpoint-best.seed2024"
        seed = 2024,
        num_samples = 512, # number of samples you want to visualize
        hl_thickness = 2.0, # the thickness of highlighting boundary of representative slots
        batch_size = 16,
        input_size = 224,
        data_set='IMNET',
        data_path = './data/in1k',
        device = 'cuda',
    )
    main(args)