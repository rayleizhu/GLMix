from argparse import Namespace
import os
from typing import Sequence

from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional as tvF
from tqdm import tqdm


def create_one_row_collage(img_list:Sequence[torch.Tensor], cut_pix_offset:int):
    imgs = torch.stack(img_list, dim=0) # requires all images have the same shape (3, h, w)
    b, _3, h, w = imgs.size()
    collage = torch.zeros(3, h, w + (w-cut_pix_offset)*(b-1),
        dtype=imgs[0].dtype, device=imgs[0].device)
    
    offset = 0
    for i, im in enumerate(imgs):
        if i == 0:
            new_offset = im.size(-1)
            collage[:, :, offset:new_offset] = im    
        else:
            new_offset = offset + im.size(-1) - cut_pix_offset
            collage[:, :, offset:new_offset] = im[:, :, cut_pix_offset:]
        offset = new_offset
    
    return collage


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    sub_dirs = args.sub_dirs
    if sub_dirs is None:
        sub_dirs = os.listdir(args.vis_root)
    assert len(sub_dirs) > 1
    files = filter(lambda x: x.endswith(".png") or x.endswith("jpg"), 
        os.listdir(os.path.join(args.vis_root, sub_dirs[0])))
    for f in tqdm(files):
        all_imgs = []
        for d in sub_dirs:
            img = Image.open(os.path.join(args.vis_root, d, f))
            img = tvF.to_tensor(img)
            # print(img.size())
            all_imgs.append(img)
        if args.onerow_mode:
            collage = create_one_row_collage(all_imgs, cut_pix_offset=args.onerow_cut_pix)
        else:
            collage = torchvision.utils.make_grid(all_imgs, nrow=args.nrow, padding=args.padding, pad_value=args.pad_value)
        collage:Image.Image = tvF.to_pil_image(collage)
        collage.save(os.path.join(args.save_dir, f))
        # break
        

if __name__ == "__main__":
    args = Namespace(
        vis_root='outputs/visualization',
        sub_dirs=('ep1-300.norm-mask.seed2024', 'ep5-300.norm-mask.seed2024', 'epbest-300.norm-mask.seed2024'),
        save_dir='outputs/vis_collage',
        onerow_mode=True, # see supp Figure 6 - 8
        onerow_cut_pix=328, # see supp Figure 6 - 8
        # below are legacy args, unused in our paper, but you may wanna try
        padding=2,
        pad_value=1,
        nrow=2,
    )
    main(args)
    