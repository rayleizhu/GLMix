import numpy as np
import torch
import torchvision.transforms.functional as tvF

from copy import deepcopy
import cv2
import torch.nn.functional as F
import io
from PIL import Image
import math

from typing import List, Sequence


import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

def blend_heatmap_and_img(heatmap:torch.Tensor, img:torch.Tensor,
        alpha:float=0.5, cmap=cv2.COLORMAP_BONE, normalize:bool=True, auto_alpha:bool=False,
        heat_intp='nearest'):
    """
    heat_map: (H, W) float tensor between [0., 1.]
    img: (C, H, W) torch.uint8 tensor
    """
    _, H, W = img.size()
    if normalize:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    if auto_alpha:
        alpha = heatmap.unsqueeze(0).unsqueeze_(0)
        alpha = F.interpolate(alpha, size=(H, W), mode=heat_intp).squeeze(0)
    heatmap_np_uint8 = np.uint8(heatmap.numpy()*255)
    heatmap_pcolored = cv2.applyColorMap(heatmap_np_uint8, cmap)
    heatmap_pcolored = cv2.resize(heatmap_pcolored, (W, H), interpolation=cv2.INTER_NEAREST)
    heatmap_pcolored = cv2.cvtColor(heatmap_pcolored, cv2.COLOR_BGR2RGB) # BGR to RGB
    heatmap_pcolored = torch.from_numpy(heatmap_pcolored).permute(2, 0, 1) #  HWC to CHW 
    blended = (1-alpha) * heatmap_pcolored + alpha * img
    return blended.to(torch.uint8)

def find_most_repsentative_ids(dist_mat, num_repr=4, max_iter=100):
    """find most repsentative ids with k-medoids
    dist_mat: N x N distance matrix

    https://zhuanlan.zhihu.com/p/55163617
    """
    # print(dist_mat)
    # print(dist_mat.min(), dist_mat.max())
    assert dist_mat.shape[0] == dist_mat.shape[1]
    n = dist_mat.shape[0]
    k = num_repr
    # Initialize the medoids randomly
    M = np.random.choice(n, k, replace=False) # Medoid 
    C = {} # Cluster dictionary

    # Run the k-medoids algorithm until convergence
    for _ in range(max_iter):
        # Assign each node to the closest medoid
        for i in range(k):
            C[i] = [] # Initialize cluster i
        for i in range(n):
            # c = np.argwhere(M == i) # (num_true, n_dim)
            # # print(type(c))
            # # print(c)
            # assert len(c) == 0 or len(c) == 1
            # if len(c) == 0:
            #     # Find the closest medoid to node i
            #     d = dist_mat[i, M] # Distance vector
            #     c = np.argmin(d) # Cluster index
            # else:
            #     c = c[0][0]
            # # Find the closest medoid to node i
            d = dist_mat[i, M] # Distance vector
            c = np.argmin(d) # Cluster index
            C[c].append(i) # Add node i to cluster c
        
        # print(M)
        # print(C)

        # Update the medoids by finding the most central node in each cluster
        M_new = np.copy(M) # New medoid indices
        for i in range(k):
            # Find the node in cluster i that minimizes the sum of distances to other nodes
            # print(len(C[i]))
            d = np.sum(dist_mat[C[i], :][:, C[i]], axis=1) # Sum of distances vector
            m = np.argmin(d) # Medoid index
            M_new[i] = C[i][m] # Update medoid i
        
        # Check for convergence
        if np.array_equal(M, M_new):
            break # Converged
        else:
            M = np.copy(M_new) # Not converged, update medoids
    
    return M

def plot_imgs(imgs, ncols=None,
    highlight_subplot_ids=None,
    highlight_colors=None,
    highlight_thickness=1.5):
    """
    plot images in a grid of axes, and highlight the axes
    with given ids (counted in row-first order)
    """

    if highlight_subplot_ids is not None:
        if highlight_colors is None:
            highlight_colors = ['red']*len(highlight_subplot_ids)
        else:
            highlight_colors = list(highlight_colors)
            assert len(highlight_subplot_ids) == len(highlight_colors)

    if not isinstance(imgs, list):
        imgs = [imgs]
    ncols = ncols or len(imgs)
    nrows = len(imgs) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = tvF.to_pil_image(img)
        ax = axs[i//ncols, i%ncols]
        ax.imshow(np.asarray(img))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        if (highlight_subplot_ids is not None) and (i in highlight_subplot_ids):
            color = highlight_colors.pop()
            for spine in ax.spines.values ():
                spine.set_linewidth(highlight_thickness)
                spine.set_edgecolor(color)

    # fig.tight_layout(pad=0.2)
    fig.set_size_inches(4, 4)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    return fig

def plt_fig_to_pil_img(fig:plt.Figure):
    buf = io.BytesIO()
    # fig.savefig(buf, format='png', pad_inches=0)
    fig.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    # when directly retun original im
    # if we close buf, there is an error when we use the image
    # otherwise, there may be memory leak
    im = deepcopy(im) 
    buf.close()
    return im

def visualize_a_batch(imgs,
        assignments:List[torch.Tensor], 
        hcolors:Sequence[str]=('red', 'orange', 'brown', 'deeppink'),
        cmap=cv2.COLORMAP_BONE,
        num_repr:int=4,
        heat_intp:str='nearest',
        highlight_thickness:float=1.5,
    )->List[Image.Image]:
    """
    imgs: (bs, 3, h, w) uint8 tensor
    assignments: a (bs, n_clusters, res, res) float tensor in the range [0, 1],
        or a list of such tensors which has length num_vis_modules
    hcolors: the hilight colors of the bounding boxes of representative visualizations
    """
    # sparsify = False
    # if sparsify:
    #     pass
    assert (hcolors is None) or ( len(hcolors)==num_repr )

    if not isinstance(assignments, (list, tuple)):
        assignments = [assignments]

    assignments = list(map(lambda x: (x - x.min()) / (x.max() - x.min()), assignments))

    vis_list = []
    # bs, n_clusters, res_h, res_w = assignments[0].size()
    bs = imgs.size(0)
    for b in range(bs):
        all_pil_imgs = []

        fig = plot_imgs([ imgs[b] ])
        pil_img = plt_fig_to_pil_img(fig)
        plt.close(fig) # suppress auto display
        # print(pil_img.size)
        all_pil_imgs.append(pil_img)

        for assignment in assignments:
            bs, n_clusters, res_h, res_w = assignment.size()

            all_ids = list(range(n_clusters))
            all_masks = [ assignment[b, j] for j in all_ids ]

            x = assignment[b].view(n_clusters, 1, res_h*res_w).expand(-1, n_clusters, -1).flatten(0, 1)
            y = assignment[b].view(1, n_clusters, res_h*res_w).expand(n_clusters, -1, -1).flatten(0, 1)
            dist_mat = F.pairwise_distance(x, y, p=2.0).view(n_clusters, n_clusters).detach().cpu().numpy()
            repr_ids = find_most_repsentative_ids(dist_mat, num_repr=num_repr, max_iter=100)
            repr_ids.sort() # small -> large ids
            repr_masks = [ assignment[b, j] for j in repr_ids ]
            repr_masks_blended = [ blend_heatmap_and_img(x, imgs[b], alpha=0.4, cmap=cmap, auto_alpha=True, heat_intp=heat_intp) for x in repr_masks ]
            
            fig2 = plot_imgs(all_masks,
                ncols=math.isqrt(n_clusters),
                highlight_subplot_ids=repr_ids,
                highlight_colors=hcolors,
                highlight_thickness=highlight_thickness)
            fig3 = plot_imgs(repr_masks_blended,
                ncols=math.isqrt(num_repr),
                highlight_subplot_ids=list(range(4)),
                highlight_colors=hcolors,
                highlight_thickness=highlight_thickness)

            for fig in (fig2, fig3):
                pil_img = plt_fig_to_pil_img(fig)
                plt.close(fig) # suppress auto display
                # print(pil_img.size)
                all_pil_imgs.append(pil_img)

        ws = [x.size[0] for x in all_pil_imgs]
        hs = [x.size[1] for x in all_pil_imgs]
        # print(ws)
        # print(sum(ws))

        grid = Image.new('RGB', (sum(ws), max(hs)))
        left_border = 0
        for x in all_pil_imgs:
            grid.paste(x, (left_border, 0))
            left_border += x.size[0]
        vis_list.append(grid)
    
    return vis_list
