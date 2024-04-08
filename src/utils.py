import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pattern_craft(im_size, pert_size):
    pert = torch.zeros(im_size)
    cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
    cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
    for c in range(im_size[0]):
        pert[c, cx, cy - 1] = pert_size
        pert[c, cx, cy + 1] = pert_size
        pert[c, cx - 1, cy] = pert_size
        pert[c, cx + 1, cy] = pert_size
        pert[c, cx, cy] = pert_size
    return pert


def add_backdoor(image, perturbation):
    image += perturbation
    image *= 255
    image = image.round()
    image /= 255
    image = image.clamp(0, 1)
    return image
