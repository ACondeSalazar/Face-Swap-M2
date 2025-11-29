""" Image utilities. """
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision

def rgb2tensor(img, normalize=True):
    """ Converts a RGB image to tensor. """
    if isinstance(img, (list, tuple)):
        return [rgb2tensor(im, normalize) for im in img]

    tensor = F.to_tensor(img)  # [0,1]
    if normalize:
        # map [0,1] -> [-1,1]
        tensor = tensor.mul(2.0).sub(1.0)

    return tensor.unsqueeze(0)

def bgr2tensor(img, normalize=True):
    """ Converts a BGR image to tensor. """
    if isinstance(img, (list, tuple)):
        return [bgr2tensor(im, normalize) for im in img]
    return rgb2tensor(img[:, :, ::-1].copy(), normalize)

def unnormalize(tensor, mean, std):
    """Undo Normalize(mean,std) that mapped to [-1,1] when mean=0.5,std=0.5."""
    # expects tensor shape (C,H,W) or (B,C,H,W)
    if tensor.dim() == 4:
        out = tensor.clone()
        for c, (m, s) in enumerate(zip(mean, std)):
            out[:, c] = out[:, c] * s + m
        return out
    else:
        out = tensor.clone()
        for c, (m, s) in enumerate(zip(mean, std)):
            out[c] = out[c] * s + m
        return out

def tensor2rgb(img_tensor):
    """ Convert an image tensor to a numpy RGB image. """
    # img_tensor can be (B,3,H,W) or (3,H,W) in [-1,1] if normalized with 0.5/0.5
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    output_img = unnormalize(img_tensor.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1]->[0,1]
    output_img = output_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img * 255).astype('uint8')
    return output_img

def tensor2bgr(img_tensor):
    """ Convert an image tensor to a numpy BGR image. """
    output_img = tensor2rgb(img_tensor)
    output_img = output_img[:, :, ::-1]
    return output_img

def make_grid(*args, cols=8):
    """ Create an image grid from a batch of images. """
    assert len(args) > 0, 'At least one input tensor must be given!'
    imgs = torch.cat([a.cpu() for a in args], dim=2)
    return torchvision.utils.make_grid(imgs, nrow=cols, normalize=True, scale_each=False)

def create_pyramid(img, n=1):
    # simple size pyramid [H,W] -> [H/2, W/2] ...
    pyr = [img]
    for _ in range(n - 1):
        pyr.append(F.resize(pyr[-1], [pyr[-1].shape[-2] // 2, pyr[-1].shape[-1] // 2]))
    return pyr