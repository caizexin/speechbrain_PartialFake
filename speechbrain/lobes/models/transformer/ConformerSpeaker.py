"""Conformer implementation.

Authors
* Jianyuan Zhong 2020
* Samuele Cornell 2021
"""

import torch
import torch.nn as nn
from typing import Optional
import speechbrain as sb
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear
import warnings


from speechbrain.nnet.attention import (
    RelPosEncXL,
    RelPosMHAXL,
    MultiheadAttention,
    PositionalwiseFeedForward,
)
from speechbrain.nnet.normalization import LayerNorm, BatchNorm1d
from speechbrain.nnet.activations import Swish
from speechbrain.dataio.dataio import length_to_mask

class ConformerSpeaker(nn.Module):
    def __init__(self, input_size, d_model, nhead, d_ffn, num_layers, embedding_size,
        CNN, pool,
        conformer_activation=Swish,
        kernel_size=31,
        attention_type="regularMHA",
        bias=True,
        causal=False,
        normalize_before=True,
        transformer_dropout=0.5,
        embedding_dropout=0.5):
        super(ConformerSpeaker, self).__init__()

        self.CNN = CNN()

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(transformer_dropout),
        )
        self.positional_encoding = RelPosEncXL(d_model)
        self.encoder = ConformerEncoder(
            nhead=nhead,
            num_layers=num_layers,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=transformer_dropout,
            activation=conformer_activation,
            kernel_size=kernel_size,
            bias=bias,
            causal=causal,
            attention_type=attention_type
        )
        assert (
            normalize_before
        ), "normalize_before must be True for Conformer"

        assert (
            conformer_activation is not None
        ), "conformer_activation must not be None"
        self.pool = pool()
        self.bottleneck = Linear(input_size=d_model*num_layers*2, n_neurons=embedding_size)
        self.norm1 = LayerNorm(input_size=d_model*num_layers)
        self.norm2 = BatchNorm1d(input_size=d_model*num_layers*2)

        self.drop = nn.Dropout(embedding_dropout) if embedding_dropout else None


    def forward(self, x, lens):
        x = self.CNN(x)
        B, T, C1, C2 = x.shape
        x = x.reshape(B, T, C1*C2)
        x = self.custom_src_module(x)

        abs_len = torch.round(lens * x.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()
        _, _, x_lst = self.encoder(x,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=self.positional_encoding(x)) # B x T x D

        x = torch.cat(x_lst, dim=-1) # B x T x nD
        x = self.norm1(x)
        x = self.pool(x.transpose(1, 2)).transpose(1, 2) # B x 1 x 2nD
        x = self.norm2(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x.squeeze(1)

class ConformerSpeakerMean(nn.Module):
    def __init__(self, input_size, d_model, nhead, d_ffn, num_layers, embedding_size,
        CNN,
        conformer_activation=Swish,
        kernel_size=31,
        attention_type="regularMHA",
        bias=True,
        causal=False,
        normalize_before=True,
        transformer_dropout=0.5,
        embedding_dropout=0.5):
        super(ConformerSpeakerMean, self).__init__()

        self.CNN = CNN()

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(transformer_dropout),
        )
        self.positional_encoding = RelPosEncXL(d_model)
        self.encoder = ConformerEncoder(
            nhead=nhead,
            num_layers=num_layers,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=transformer_dropout,
            activation=conformer_activation,
            kernel_size=kernel_size,
            bias=bias,
            causal=causal,
            attention_type=attention_type
        )
        assert (
            normalize_before
        ), "normalize_before must be True for Conformer"

        assert (
            conformer_activation is not None
        ), "conformer_activation must not be None"
        self.bottleneck = Linear(input_size=d_model*num_layers, n_neurons=embedding_size)
        self.norm1 = LayerNorm(input_size=d_model*num_layers)
        self.norm2 = BatchNorm1d(input_size=d_model*num_layers)

        self.drop = nn.Dropout(embedding_dropout) if embedding_dropout else None


    def forward(self, x, lens):
        x = self.CNN(x)
        B, T, C1, C2 = x.shape
        x = x.reshape(B, T, C1*C2)
        x = self.custom_src_module(x)

        abs_len = torch.round(lens * x.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()
        _, _, x_lst = self.encoder(x,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=self.positional_encoding(x)) # B x T x D

        x = torch.cat(x_lst, dim=-1) # B x T x nD
        x = self.norm2(x)
        x = self.bottleneck(x) # B x T x D
        x = x.mean(1) # B x D
        if self.drop:
            x = self.drop(x)
        return x.squeeze(1)

class ConformerSpeakerFrame(nn.Module):
    def __init__(self, input_size, d_model, nhead, d_ffn, num_layers, embedding_size,
        CNN,
        conformer_activation=Swish,
        kernel_size=31,
        attention_type="regularMHA",
        bias=True,
        causal=False,
        normalize_before=True,
        transformer_dropout=0.5,
        embedding_dropout=0.5):
        super(ConformerSpeakerFrame, self).__init__()

        self.CNN = CNN()

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(transformer_dropout),
        )
        self.positional_encoding = RelPosEncXL(d_model)
        self.encoder = ConformerEncoder(
            nhead=nhead,
            num_layers=num_layers,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=transformer_dropout,
            activation=conformer_activation,
            kernel_size=kernel_size,
            bias=bias,
            causal=causal,
            attention_type=attention_type
        )
        assert (
            normalize_before
        ), "normalize_before must be True for Conformer"

        assert (
            conformer_activation is not None
        ), "conformer_activation must not be None"
        self.bottleneck1 = Linear(input_size=d_model*num_layers, n_neurons=embedding_size)
        self.bottleneck2 = Linear(input_size=d_model*num_layers, n_neurons=embedding_size)
        self.norm1 = LayerNorm(input_size=d_model*num_layers)
        self.norm2 = BatchNorm1d(input_size=d_model*num_layers)

    def forward(self, x, lens):
        x = self.CNN(x)
        B, T, C1, C2 = x.shape
        x = x.reshape(B, T, C1*C2)
        x = self.custom_src_module(x)

        abs_len = torch.round(lens * x.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()
        _, _, x_lst = self.encoder(x,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=self.positional_encoding(x)) # B x T x D

        x = torch.cat(x_lst, dim=-1) # B x T x nD
        x = self.norm2(x)
        x1 = self.bottleneck1(x) # B x T x D
        x2 = self.bottleneck2(x) # B x T x D

        return x1, x2

class ConvolutionModule(nn.Module):
    """This is an implementation of convolution module in Conformer.

    Arguments
    ----------
    input_size : int
        The expected size of the input embedding dimension.
    kernel_size: int, optional
        Kernel size of non-bottleneck convolutional layer.
    bias: bool, optional
        Whether to use bias in the non-bottleneck conv layer.
    activation: torch.nn.Module
         Activation function used after non-bottleneck conv layer.
    dropout: float, optional
         Dropout rate.
    causal: bool, optional
         Whether the convolution should be causal or not.
    dilation: int, optional
         Dilation factor for the non bottleneck conv layer.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = ConvolutionModule(512, 3)
    >>> output = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        input_size,
        kernel_size=31,
        bias=True,
        activation=Swish,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.causal = causal

        if self.causal:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1)
        else:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1) // 2

        self.layer_norm = nn.LayerNorm(input_size)
        self.bottleneck = nn.Sequential(
            # pointwise
            nn.Conv1d(
                input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias
            ),
            nn.GLU(dim=1),
        )
        # depthwise
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
            groups=input_size,
            bias=bias,
        )

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            activation(),
            # pointwise
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """ Processes the input tensor x and returns the output an output tensor"""
        out = self.layer_norm(x)
        out = out.transpose(1, 2)
        out = self.bottleneck(out)
        out = self.conv(out)

        if self.causal:
            # chomp
            out = out[..., : -self.padding]
        out = out.transpose(1, 2)
        out = self.after_conv(out)
        if mask is not None:
            out.masked_fill_(mask, 0.0)
        return out


class ConformerEncoderLayer(nn.Module):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ----------
    d_model : int
        The expected size of the input embedding.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: torch.nn.Module
         Activation function used in each Conformer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_embs = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoderLayer(d_ffn=512, nhead=8, d_model=512, kernel_size=3)
    >>> output = net(x, pos_embs=pos_embs)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
        elif attention_type == "RelPosMHAXL":
            # transformerXL style positional encoding
            self.mha_layer = RelPosMHAXL(
                num_heads=nhead,
                embed_dim=d_model,
                dropout=dropout,
                mask_pos_future=causal,
            )

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the input sequence positional embeddings
        """
        conv_mask = None
        if src_key_padding_mask is not None:
            conv_mask = src_key_padding_mask.unsqueeze(-1)
        # ffn module
        x = x + 0.5 * self.ffn_module1(x)
        # muti-head attention module
        skip = x
        x = self.norm1(x)
        x, self_attn = self.mha_layer(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )
        x = x + skip
        # convolution module
        x = x + self.convolution_module(x, conv_mask)
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn


class ConformerEncoder(nn.Module):
    """This class implements the Conformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Embedding dimension size.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: torch.nn.Module
         Activation function used in each Confomer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.


    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_emb = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoder(1, 512, 512, 8)
    >>> output, _ = net(x, pos_embs=pos_emb)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the input sequence positional embeddings
        """
        output = src
        attention_lst = []
        output_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
            )
            attention_lst.append(attention)
            output_lst.append(output)
        output = self.norm(output)

        return output, attention_lst, output_lst


