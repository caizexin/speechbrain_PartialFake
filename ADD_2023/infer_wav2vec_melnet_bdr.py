#!/usr/bin/python3
"""
Recipe for infering x-vector-based target speaker voice activity detection
"""

import os, csv, sys, time, tqdm, glob
import random, itertools
from collections import defaultdict

import scipy.io.wavfile as wf
from python_speech_features import sigproc
import numpy as np #
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torch, torchaudio
import torch.nn as nn, torch.nn.functional as F
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.diarization_utils import rttm2label_dict, label2segments
from speechbrain.utils import checkpoints
from utils import EER
from dataset import valid_dataio_prep_bdr as valid_dataio_prep, valid_dataio_addtest_bdr, valid_dataio_prep_bdr_wfpath_add2022
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
def compute_decision(mixtures):
    with torch.no_grad():
        feats = hparams["wav2vec"](mixtures)
        embds = hparams["front"](feats)
        out = hparams["back"](embds) # B x T x n_spk
    return out

def compute_decision_loop(hparams, dataloader, add_noise=False, need_label=True):
    label_length = round(hparams["wav_length"] / hparams["time_resolution"])
    
    predictions, labels = [], []
    if not need_label:
        labels = None
    wavs = []

    for batch in tqdm.tqdm(dataloader, dynamic_ncols=True):
        target, _      = batch.target
        mixtures, lens = batch.sig
        label = batch.b_target
        lab_starts = batch.lab_starts[0]
        lab_stops = batch.lab_stops[0]
        wav   = batch.wav

        pred = torch.zeros(label_length * (mixtures.shape[1]+1))
        count = torch.zeros(label_length * (mixtures.shape[1]+1))

        mixtures = mixtures[0]
        mixtures = mixtures.to(hparams["device"])

        if add_noise:
            lens = torch.tensor([mixtures.shape[0]]).long().to(hparams["device"])
            mixtures = hparams["add_noise"](mixtures, lens)

        if mixtures.shape[0] > 50:
            half_idx = mixtures.shape[0] // 2
            decision1 = compute_decision(mixtures[:half_idx, :])
            decision2 = compute_decision(mixtures[half_idx:, :])
            decision = torch.cat((decision1, decision2), 0)
        else:
            decision = compute_decision(mixtures)
        decision = torch.sigmoid(decision).detach().clone().cpu().squeeze(-1)

        for i, d in enumerate(decision):
            s = lab_starts[i]
            e = lab_stops[i]
            if e - s > d.shape[0]:
                new_d = np.zeros(d.shape[0] + 1)
                new_d[:-1] = d
                new_d[-1] = d[-1]
                d = new_d
            if e - s < d.shape[0]:
                new_d = np.zeros(d.shape[0] - 1)
                new_d = d[:-1]
                d = new_d

            pred[s:e] += d
            count[s:e] += 1.

        pred = pred[:e] / count[:e]
        assert e >= target[0].shape[0]
        pred = pred[:target[0].shape[0]]

        predictions.append(pred)
        if need_label:
            labels.append(label[0][0])
        wavs.append(wav[0])

    return predictions, labels, wavs


def get_results(dataloader, hp, testset="", add_noise=False, evaluate=True, need_label=True):
    print("============ %s ============" % ("Infering "+testset))

    eval_dataloader = sb.dataio.dataloader.make_dataloader(
        dataloader, **hp["dataloader_options"])
    preds, trues, wavs = compute_decision_loop(hparams, eval_dataloader, add_noise, need_label)

    if evaluate:
        labels, scores = [], []
        for i in range(len(preds)):
            labels.append(trues[i].detach().cpu().numpy())
            scores.append(1-preds[i].sort()[0].detach().cpu().numpy()[:4].mean())
        eer = EER(labels, scores)

        print("EER", eer)
    return preds, trues, wavs

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

    # load pretrained model
    print(hparams["checkpoint_save_folder"])
    best_ckpt = hparams["checkpointer"].find_checkpoints(min_key="EER")

    # merge the top 5 models with the lowest EER for inference
    ckpts = hparams["checkpointer"].find_checkpoints(
            min_key="EER", max_num_checkpoints=5
        )
    print([i.path for i in ckpts])

    front_ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="front", device="cpu"
    )
    back_ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="back", device="cpu"
    )
    wav2vec_ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="wav2vec", device="cpu"
    )
    avg_ckpt = {
        "front": front_ckpt,
        "back": back_ckpt,
        "wav2vec": wav2vec_ckpt
    }
    #print("Load checkpoint from: ", best_ckpt.path)
    for model in ["back", "front", "wav2vec"]:
        hparams[model].load_state_dict(avg_ckpt[model], strict=True)
        hparams[model].eval()
        hparams[model].to(hparams["device"])
    #hparams["compute_features"].to(hparams["device"])
    hparams["add_noise"].to(hparams["device"])
    os.makedirs(hparams["inference_save_folder"], exist_ok=True)

    # infer add 2023 Dev
    adapt_dataloader = valid_dataio_prep(hparams)
    # # preds, trues, wavs = get_results(adapt_dataloader, hparams, "ADD 2023 dev with 15db noise", True)
    preds, trues, wavs = get_results(adapt_dataloader, hparams, "ADD 2023 dev") 

    # Infer ADD 2023 Test
    # adapt_dataloader = valid_dataio_addtest_bdr(hparams)
    # preds, trues, wavs = get_results(adapt_dataloader, hparams, "ADD 2023 test", evaluate=False) 
    # wav2pred = {wav:pred for wav, pred in zip(wavs, preds)}
    # torch.save(wav2pred, "%s/add2023_wav2preds%d.arr" % (hparams["inference_save_folder"], hparams["i_chunks"]))

    #add_2022_path = '/work/caizexin/ADD/track2test/'
    #add_2022_file = 'add2022_sorted_wav_list'
    #add_2022_worldresyn_dataloader = valid_dataio_prep_bdr_wfpath_add2022(hparams, add_2022_file, add_2022_path)
    #preds, trues, wavs = get_results(add_2022_worldresyn_dataloader, hparams, "ADD 2022 Test", evaluate=False, need_label=False) 
    #wav2pred = {wav:pred for wav, pred in zip(wavs, preds)}

    #torch.save(wav2pred, "%s/add2022_wav2preds%d.arr" % (hparams["inference_save_folder"], hparams["i_chunks"]))
    #with open("%s/scores.txt" % (hparams["inference_save_folder"]), 'w') as f:
    #    for wav, pred in wav2pred.items():
    #        f.write("%s.wav %lf\n" % (wav, pred.sort()[0].detach().cpu().numpy()[:4].mean()))
