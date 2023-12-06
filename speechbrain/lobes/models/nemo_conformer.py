"""This lobe enables the integration of huggingface pretrained wav2vec2/hubert/wavlm models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2110.13900
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
"""

import os
import torch
import logging
import pathlib
import numpy as np
import torch.nn.functional as F
from torch import nn

import nemo.collections.asr as nemo_asr
from speechbrain.pretrained.fetching import fetch

logger = logging.getLogger(__name__)


class NemoConformer(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained wav2vec2.0/Hubert models.

    Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Source paper Hubert: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    output_all_hiddens : bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example wav2vec2-base has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        model_name,
        output_norm=True,
        freeze=True,
        normalize = False,
        freeze_feature_extractor=False
    ):
        super().__init__()

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation information
        logger.warning(
                "Loading pretrain model " + model_name 
            )
        self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)
        self.preprocessor = self.asr_model.preprocessor
        self.model = self.asr_model.encoder
        
        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = normalize

        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_norm = output_norm

        if self.freeze:
            logger.warning(
                "pretrain model from nemo is frozen."
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()

    def forward(self, feat, feat_lengths):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(feat, feat_lengths).detach()

        return self.extract_features(feat, feat_lengths)

    def extract_features(self, feat, feat_lengths):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # if self.normalize_wav:
        #     wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        out, _ = self.model(audio_signal=feat, length=feat_lengths)

        norm_shape = out.shape

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, norm_shape)

        return out


