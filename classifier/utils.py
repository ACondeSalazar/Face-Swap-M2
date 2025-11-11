import torch

# https://perso.esiee.fr/~chierchg/deep-learning/tutorials/metric/metric-2.html
def get_pairs(labels):
    positive = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive.fill_diagonal_(False)
    negative = labels.unsqueeze(0) != labels.unsqueeze(1)
    return positive, negative

#pour chaque element de label, dit si les autres images ont le meme label (pos), ou sont different (neg)
def get_triplets(labels):
    pos, neg = get_pairs(labels)
    pos = pos.unsqueeze(2)
    neg = neg.unsqueeze(1)

    triplets = pos & neg
    return torch.nonzero(triplets, as_tuple=True)