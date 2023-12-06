#!/usr/bin/python3
"""
Recipe for training x-vector-based target speaker voice activity detection
"""

import os, csv, sys, time, tqdm, glob
import random, itertools
from collections import defaultdict

import numpy as np #
import scipy.io.wavfile as wf
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from python_speech_features import sigproc
from dataset import train_dataio_prep_bdr_dc as train_dataio_prep
from dataset import valid_dataio_prep_bdr as valid_dataio_prep
import torch, torchaudio, torch.nn.functional as F
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils import checkpoints
from utils import EER, f1score

class ADDBrain(sb.core.Brain):
    """
    class for similarity measurement training
    """
    def init_optimizers(self):
        if self.opt_class is not None:
            self.optimizer = self.opt_class(self.modules.parameters())
            # do not save or load optimizer here

    def compute_forward(self, batch, stage):
        """
        Computation pipeline
        """
        mixtures, lens = batch.sig
        labels, _      = batch.target
        mixtures = mixtures.to(self.device)
        lens = lens.to(self.device)

        if stage == sb.Stage.TRAIN:
            augment = random.choice(self.hparams.augment_pipeline)
            mixtures = augment(mixtures, lens)

        if stage == sb.Stage.VALID:
            mixtures = mixtures[0]

        if self.hparams.freeze_wav2vec:
            with torch.no_grad():
                feats = self.hparams.wav2vec(mixtures)
        else:
            feats = self.hparams.wav2vec(mixtures) 

        embds = self.hparams.front(feats)
        out = self.modules.back(embds)

        return embds, out

    def compute_objectives(self, embeds, predictions, batch, stage):
        """Computes the loss
        """
        labels, _      = batch.target # B x T x 1
        labels = labels.to(self.device)[:, :-1, :]
        labels = 1 - labels
        _, _, D = embeds.shape
        loss = self.hparams.compute_cost(predictions, labels)

        return loss

    def fit_batch(self, batch):
        #self.check_and_reset_optimizer()

        embeds, predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(embeds, predictions, batch, sb.Stage.TRAIN)

        (loss / self.hparams.gradient_accumulation).backward()
        if self.step % self.hparams.gradient_accumulation == 0:
            self.check_gradients(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        label_length = round(hparams["wav_length"] / hparams["time_resolution"])
        
        labels, _      = batch.target
        blabels, _      = batch.b_target
        mixtures, _ = batch.sig
        labels = 1 - labels
        blabels = 1 - blabels

        pred = torch.zeros(label_length * (mixtures.shape[1]+1))
        count = torch.zeros(label_length * (mixtures.shape[1]+1))

        _, decision = self.compute_forward(batch, stage=stage)
        
        # decision = torch.sigmoid(decision).detach().clone().cpu().squeeze(-1).numpy()
        decision = torch.sigmoid(decision).detach().clone().cpu().squeeze(-1)

        lab_starts = batch.lab_starts[0]
        lab_stops = batch.lab_stops[0]

        for i, d in enumerate(decision):
            s = lab_starts[i]
            e = lab_stops[i]
            if e - s > d.shape[0]:
                new_d = np.ones(d.shape[0] + 1)
                new_d[:-1] = d
                new_d[-1] = d[-1]
                d = new_d
            if e - s < d.shape[0]:
                new_d = np.ones(d.shape[0] - 1)
                new_d = d[:-1]
                d = new_d

            pred[s:e] += d
            count[s:e] += 1.
    
        pred = pred[:e] / count[:e]
        pred = pred[:labels[0].shape[0]]

        self.valid_ids.append(batch.id)
        self.valid_scores.append(pred.sort()[0].detach().cpu().numpy()[:4].mean())
        self.valid_preds.append(torch.round(pred))
        self.valid_labels.append(labels[0])
        self.valid_blabels.append(blabels[0][0])
        # no loss here since we do not have a label for each frame
        return torch.tensor([0])

    def on_stage_start(self, stage, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.step = 0
        if stage != sb.Stage.TRAIN:
            self.valid_metrics = self.hparams.error_stats()
            self.valid_ids = []
            self.valid_preds = []
            self.valid_scores = []
            self.valid_labels = []
            self.valid_blabels = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            for i in range(len(self.valid_labels)):
                if i == 0:
                    seg_labels = self.valid_labels[i].detach().cpu().numpy()
                    preds = self.valid_preds[i].detach().cpu().numpy()
                else:
                    seg_labels = np.concatenate((seg_labels, self.valid_labels[i].detach().cpu().numpy()))
                    preds = np.concatenate((preds, self.valid_preds[i].detach().cpu().numpy()))
                    
            stage_stats["f1score"] = f1score(seg_labels, preds)
            stage_stats["EER"] = EER(torch.tensor(self.valid_blabels).numpy(), torch.tensor(self.valid_scores).numpy())

        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                num_to_keep=self.hparams.n_checkpoints,
                min_keys=["EER"],
                name="epoch_{}".format(epoch),
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


if __name__ == "__main__":
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare Musan
    from data_prep.musan_prepare import prepare_musan
    run_on_main(
        prepare_musan,
        kwargs={
            "folder": hparams["musan_folder"],
            "music_csv": hparams["music_csv"],
            "noise_csv": hparams["noise_csv"],
            "speech_csv": hparams["speech_csv"],
        },
    )

    # Brain class initialization
    train_data = train_dataio_prep(hparams)
    valid_data = valid_dataio_prep(hparams)
    add_brain = ADDBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # Training
    add_brain.fit(
        add_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )
