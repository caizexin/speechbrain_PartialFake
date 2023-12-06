"""
Several front- and back-end models for End-to-end neural speaker diarization

Authors
 * Weiqing Wang 2020
"""
import torch, torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder
from speechbrain.lobes.models.longformer.longformer import LongformerEncoder
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear

class EEND_TransformerEncoder(nn.Module):
    def __init__(self,
            # Transformer Parameters
            num_layers=4, nhead=8, d_ffn=256, d_model=128, dropout=0.1,
            # LSTM parameters
            hidden_size=128,
            # fc Parameter
            n_spk=4):
        torch.nn.Module.__init__(self)

        self.encoder = TransformerEncoder(
            num_layers, nhead, d_ffn,
            d_model=d_model,
            dropout=dropout
        )
        self.classifier = nn.Linear(d_model, n_spk)
        self.lstm = nn.LSTM(d_model, hidden_size, 1, batch_first=True, bidirectional=True, dropout=0)
        self.back  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size*2, n_spk)
        )

    def forward(self, x):
        x, att = self.encoder(x)
        x, _ = self.lstm(x)
        x = self.back(x)
        return x
    
class ADD_LongformerEncoder(nn.Module):
    def __init__(self,
            # Transformer Parameters
            num_layers=4, nhead=8, d_ffn=256, d_model=128, dropout=0.1,
            attention_window=5,
            # LSTM parameters
            hidden_size=128,
            # fc Parameter
            n_spk=4):
        torch.nn.Module.__init__(self)

        self.encoder = LongformerEncoder(
            num_layers, nhead, d_ffn,
            d_model=d_model,
            attention_window=attention_window,
            dropout=dropout
        )
        self.classifier = nn.Linear(d_model, n_spk)
        self.lstm = nn.LSTM(d_model, hidden_size, 1, batch_first=True, bidirectional=True, dropout=0)
        self.back  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size*2, n_spk)
        )

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.lstm(x)
        x = self.back(x)
        return x

class EEND_BiLSTM(nn.Module):
    def __init__(self,
            input_size=128, num_layers=2, dropout=0.1, hidden_size=128,
            # fc Parameter
            fc_size=128, n_spk=4):
        torch.nn.Module.__init__(self)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.back  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size*2, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, n_spk)
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.back(x)
        return x

class ECAPA_Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.
    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.
    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outupts = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_blocks=0,
        lin_neurons=192,
        out_neurons=1211,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    _BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Returns the output probabilities over speakers.
        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.
        """
        for layer in self.blocks:
            x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)

class EEND_TransformerEncoder_aam(nn.Module):
    def __init__(self,
            # Transformer Parameters
            num_layers=4, nhead=8, d_ffn=256, d_model=128, dropout=0.1,
            # LSTM parameters
            hidden_size=128,
            # fc Parameter
            n_spk=4):
        torch.nn.Module.__init__(self)

        self.encoder = TransformerEncoder(
            num_layers, nhead, d_ffn,
            d_model=d_model,
            dropout=dropout
        )

        self.classifier = ECAPA_Classifier(d_model, out_neurons=n_spk)
        self.lstm = nn.LSTM(d_model, hidden_size, 1, batch_first=True, bidirectional=True, dropout=0)
        self.back  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size*2, d_model)
        )

    def forward(self, x):
        x, att = self.encoder(x)
        x, _ = self.lstm(x)
        x = self.back(x)
        _, _, D = x.shape
        x = x.reshape(-1, D).unsqueeze(1)
        x = self.classifier(x)
        return x
