import torch.nn as nn
from model.layers.img_encoder import ResnetEncoder
from model.layers.sentence_encoder import make_encoder


class TripletMatch(nn.Module):
    def __init__(self, vec_dim=256, distance='l2_s', img_feats=(2, 4)):
        super(TripletMatch, self).__init__()
        self.vec_dim = vec_dim

        if distance == 'l2_s':
            self.dist_fn = lambda v1, v2: (v1 - v2).pow(2).sum(dim=-1)
        elif distance == 'l2':
            self.dist_fn = lambda v1, v2: (v1 - v2).pow(2).sum(dim=-1).sqrt()
        elif distance == 'cos':
            self.dist_fn = lambda v1, v2: 1.0 - nn.functional.cosine_similarity(v1, v2, dim=0)
        else:
            raise NotImplementedError

        self.resnet_encoder = ResnetEncoder(use_feats=img_feats)
        self.lang_embed = make_encoder()
        self.img_encoder = nn.Sequential(self.resnet_encoder, nn.Linear(self.resnet_encoder.out_dim, vec_dim))
        self.lang_encoder = nn.Sequential(self.lang_embed, nn.Linear(self.lang_embed.out_dim, vec_dim))
        return

