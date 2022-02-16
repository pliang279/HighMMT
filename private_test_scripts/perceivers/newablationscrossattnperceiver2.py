from typing import Iterable, Dict, List

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch import nn
from torch.nn import Identity

from perceiver_pytorch.caching import cache_by_name_fn
from perceiver_pytorch.modalities import InputModality, modality_encoding
from perceiver_pytorch.perceiver_pytorch import PreNorm, Attention, FeedForward, cache_fn, fourier_encode, \
    FeedForwardGELU
from perceiver_pytorch.common import build_perceiver_layers

def modality_encoding(batch_size: int, axes, modality_index: int, num_modalities: int, embed=None,
                      device=torch.device('cpu')) -> Tensor:
    """
    Return one-hot encoding of modality given num_modalities, batch size and axes.
    The result need to be compatible with the modality data for concatenation.
    :param modality_index:
    :param num_modalities:
    :return:
    """
    #modality_index=0
    if embed is None:
        one_hot = torch.eye(num_modalities, num_modalities, device=device)[modality_index]
    else:
        one_hot=embed[modality_index]
    to_expand = [batch_size]
    one_hot = one_hot.unsqueeze(0)
    for i, axis in enumerate(axes):
        one_hot = one_hot.unsqueeze(0)
        to_expand.append(axis)
    if embed is None:
        to_expand.append(num_modalities)
    else:
        to_expand.append(len(embed[0]))

    one_hot = one_hot.expand(to_expand)
    return one_hot



def findmodalityandindex(ms,mn):
    for i,m in enumerate(ms):
        if mn == m.name:
            return m,i

# An implementation of Perceiver that can accept multiple data modalities in the same forward.
class MultiModalityPerceiver(nn.Module):
    def __init__(
            self,
            *,
            modalities: Iterable[InputModality],
            depth,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=None,
            attn_dropout=0.,
            ff_dropout=0.,
            embed=False,
            embed_size=10,
            weight_tie_layers=False,
            num_latent_blocks_per_layer=1,
            use_gelu: bool = False,
            cross_depth=2,
            cross_cross_heads=4,
            recon=None
    ):
        """
        :param modalities:
        :param depth: Number of times the perceiver will perform cross-attention between latent and input.
        :param num_latents:
        :param latent_dim:
        :param cross_heads:
        :param latent_heads:
        :param cross_dim_head:
        :param latent_dim_head:
        :param num_classes: Number of classes to predict, or if None, return the hidden state (num latents x hidden_dim)
        :param attn_dropout:
        :param ff_dropout:
        :param weight_tie_layers: True: share weights across layers, False no shared weights.
        :param num_latent_blocks_per_layer: Number of blocks in the latent transformer.
        :param use_gelu: Use GELU activation like the Perceiver preprint indicates. False,
               with Lucidrains' GEGLU activation in feed forward instead.
        """
        super().__init__()
        self.modalities = modalities
        self.embed_size=embed_size
        # we encode modality with one hot encoding, so need one dim per modality:
        modality_encoding_dim = sum([1 for _ in modalities])
        nummodalities = modality_encoding_dim
        if embed:
            modality_encoding_dim=embed_size
        self.modality_encoding_dim=modality_encoding_dim
        # input_dim is the maximum dimension over all input modalities:
        input_dim = max(modality.input_dim for modality in modalities) + modality_encoding_dim
        self.max_modality_dim = input_dim
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        ff_type = FeedForwardGELU if use_gelu else FeedForward
        self.embed=None
        if embed:
            self.embed = torch.nn.Parameter(torch.randn(nummodalities,embed_size))
        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head,
                                                   dropout=attn_dropout), context_dim=input_dim)
        get_cross_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, latent_dim, heads=cross_cross_heads, dim_head=cross_dim_head,
                                                   dropout=attn_dropout), context_dim=latent_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, ff_type(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, ff_type(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_by_name_fn, (
            get_cross_attn,get_cross_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])

        build_perceiver_layers(self.layers, depth, get_cross_attn, get_cross_ff,
                               get_latent_attn, get_latent_ff,
                               weight_tie_layers,
                               num_latent_blocks_per_layer=num_latent_blocks_per_layer)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim*2),
            nn.Linear(latent_dim*2, num_classes)
        )
        
        self.cross_layers = []
        for i in range(4):
            layes=nn.ModuleList([])
            build_perceiver_layers(layes, cross_depth, get_cross_cross_attn, get_cross_ff,
                               get_latent_attn, get_latent_ff,
                               weight_tie_layers,
                               num_latent_blocks_per_layer=num_latent_blocks_per_layer)
            self.cross_layers.append(layes)
        self.cross_layers=nn.ModuleList(self.cross_layers)
        self.recon=recon

    def forward(self, multi_modality_data: Dict[str, Tensor], mask=None, use_recon=False,tasknum=-1):
        """
        :param data: a dictionary where keys are modality names and Tensor contain a batch
        of modality input data.
        :param mask:
        :return:
        """
        batch_sizes = set()
        num_modalities = len(self.modalities)
        linearized_data = []
        linearized_data_per_layer: Dict[int, List[Tensor]] = {}
        latentout=[]
        for _, modality_name in enumerate(sorted(multi_modality_data.keys())):
            #assert modality_name in self.modalities, f"modality {modality_name} was not defined in constructor"
            data = multi_modality_data[modality_name]
            modality,modality_index = findmodalityandindex(self.modalities,modality_name)
            #print(data.shape)

            b, *axis, _, device = *data.shape, data.device
            assert len(
                axis) == modality.input_axis, f'input data must have the right number of  for modality {modality_name}. ' \
                                              f'Expected {modality.input_axis} while forward argument offered {len(axis)}'
            batch_sizes.add(b)
            assert len(batch_sizes) == 1, "batch size must be the same across all modalities"
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos,
                                     modality.max_freq, modality.num_freq_bands, modality.freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            # Figure out padding for this modality, given max dimension across all modalities:
            padding_size = self.max_modality_dim - modality.input_dim - self.modality_encoding_dim

            padding = torch.zeros(size=data.size()[0:-1] + (padding_size,)).to(device)
            # concat to channels of data and flatten axis
            modality_encodings = modality_encoding(b, axis, modality_index, num_modalities, embed=self.embed, device=device)

            #print(modality_encodings.size())

            to_concat = (data, padding, enc_pos, modality_encodings)


            data = torch.cat(to_concat, dim=-1)
            #print(data.size())
            data = rearrange(data, 'b ... d -> b (...) d')
            #print(data.size())
            #linearized_data.append(data)
        
            b = batch_sizes.pop()
            x = repeat(self.latents, 'n d -> b n d', b=b)
        
            # Concatenate all the modalities:
            #data = torch.cat(linearized_data, dim=1)

            for cross_attn, cross_ff, latent_transformer in self.layers:
                x = cross_attn(x, context=data, mask=mask) + x
                x = cross_ff(x) + x
                x = latent_transformer(x) + x
            #x = self.pool(x)
            latentout.append(x)

        outs=[]
        for i in range(len(latentout)):
            for j in range(len(latentout)):
                if i==j:
                    continue
                x=latentout[i]
                context=latentout[j]
                for cross_attn, cross_ff, latent_transformer in self.cross_layers[tasknum]:
                    x = cross_attn(x, context=context, mask=mask) + x
                    x = cross_ff(x) + x
                    x = latent_transformer(x) + x
                outs.append(x[:,-1])


        catted=torch.cat(outs,dim=1)
        if (self.recon is not None) and use_recon:
            return self.to_logits(catted),self.recon(catted)
        return self.to_logits(catted)

