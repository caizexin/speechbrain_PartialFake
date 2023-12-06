import os, csv, sys, time, tqdm, glob
import random, itertools
from collections import defaultdict

import numpy as np
import speechbrain as sb
import torch, torchaudio

def train_dataio_prep_bdr(hparams):
    wav_length = int(hparams["wav_length"] * hparams["sample_rate"])
    sample_rate = int(hparams["sample_rate"])
    label_length = int(hparams["wav_length"] / hparams["time_resolution"])
    train_wav2info = sorted([line.strip().split() for line in open(hparams["add_train"])])
    if 'wav2vec' in hparams["feature"] or 'wavLM' in hparams["feature"]:
        wav_length = int((hparams["wav_length"]+hparams["time_resolution"]) * hparams["sample_rate"])
        label_length = int(hparams["wav_length"] / hparams["time_resolution"]) + 1
    train_dict = {}
    wav2sig = {}
    nobdr_sigs = []
    # training samples

    for i in tqdm.tqdm(range(len(train_wav2info)), ncols=64):
        wav, info, blable = train_wav2info[i][0], train_wav2info[i][1], train_wav2info[i][2]
        y = sb.dataio.dataio.read_audio(hparams["train_wav_folder"]+"/wav/"+wav+'.wav')
        wav2sig[wav] = y
        real_audio_len = y.shape[0]
        utt = wav# remove .wav
        info = info.split('/') # each element is start-end-label
        labels = [1 if l.split('-')[-1] == 'T' else 0 for l in info]
        segs = [[int(float(j.split('-')[0]) * sample_rate), int(float(j.split('-')[1]) * sample_rate)] for j in info]
        
        if len(labels) == 1:
            if y.shape[0] > wav_length:
                nobdr_sigs.append(y)
            continue
        
        if segs[-1][-1] > wav_length:
            segs[-1][-1] = real_audio_len
            assert segs[-1][0] < segs[-1][1]
            train_dict[utt] = {
                "wav": wav,
                "segs": segs,
                "labels": labels
            }

    print(len(train_dict))
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset(train_dict)
    datasets = [train_data]

    # 2. Define pipeline:
    @sb.utils.data_pipeline.takes("wav", "segs", "labels")
    @sb.utils.data_pipeline.provides("sig", "target")
    def audio_pipeline(wav, segs, labels):
        sr = int(hparams["sample_rate"])


        if random.random() < hparams["positive_ratio"]: # positive samples
            #sr, y = wf.read(random.choice(genuine_wav_list))
            y = random.choice(nobdr_sigs)
            if len(y) - wav_length < 0:
                print(y.shape, wav)
            start = random.randint(0, len(y) - wav_length - 1)
            stop  = start + wav_length
            sig = y[start:stop]
            target = torch.zeros((label_length, 1))
        else:
            y = wav2sig[wav]
            if len(y) - wav_length < 0:
                print(y.shape, wav)

            start = random.randint(0, len(y) - wav_length - 1)
            stop  = start + wav_length
            sig = y[start:stop]

            change_points = [start for start, stop in segs[1:]]
            target = torch.zeros((label_length, 1))
            
            for cp in change_points:
                if start < cp < stop:
                    idx = round((cp - start) / hparams["sample_rate"] / hparams["time_resolution"])
                    lab_start = idx - hparams["neighborhood_size"]
                    if lab_start < 0:
                        lab_start = 0
                    lab_stop  = idx + hparams["neighborhood_size"]
                    if lab_stop > label_length:
                        lab_stop = label_length
                    target[lab_start:lab_stop] = 1

        return sig.float(), target.float()

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "target"])

    return train_data

def train_dataio_prep_bdr_dc(hparams):
    # with dynamic online concatenation 
    wav_length = int(hparams["wav_length"] * hparams["sample_rate"])
    sample_rate = int(hparams["sample_rate"])
    label_length = int(hparams["wav_length"] / hparams["time_resolution"])
    train_wav2info = sorted([line.strip().split() for line in open(hparams["add_train"])])
    if 'wav2vec' in hparams["feature"] or 'wavLM' in hparams["feature"]:
        wav_length = int((hparams["wav_length"]+hparams["time_resolution"]) * hparams["sample_rate"])
        label_length = int(hparams["wav_length"] / hparams["time_resolution"]) + 1
    train_dict = {}
    wav2sig = {}
    nobdr_sigs = []
    # training samples

    for i in tqdm.tqdm(range(len(train_wav2info)), ncols=64):
        wav, info, blable = train_wav2info[i][0], train_wav2info[i][1], train_wav2info[i][2]
        y = sb.dataio.dataio.read_audio(hparams["train_wav_folder"]+"/wav/"+wav+'.wav')
        wav2sig[wav] = y
        real_audio_len = y.shape[0]
        utt = wav# remove .wav
        info = info.split('/') # each element is start-end-label
        labels = [1 if l.split('-')[-1] == 'T' else 0 for l in info]
        segs = [[int(float(j.split('-')[0]) * sample_rate), int(float(j.split('-')[1]) * sample_rate)] for j in info]
        
        if len(labels) == 1:
            if y.shape[0] > wav_length:
                nobdr_sigs.append(y)
            continue
        
        if segs[-1][-1] > wav_length:
            segs[-1][-1] = real_audio_len
            assert segs[-1][0] < segs[-1][1]
            train_dict[utt] = {
                "wav": wav,
                "segs": segs,
                "labels": labels
            }

    print(len(train_dict))
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset(train_dict)
    datasets = [train_data]

    # 2. Define pipeline:
    @sb.utils.data_pipeline.takes("wav", "segs", "labels")
    @sb.utils.data_pipeline.provides("sig", "target")
    def audio_pipeline(wav, segs, labels):
        sr = int(hparams["sample_rate"])

        if random.random() < hparams["positive_ratio"]: # positive samples
            y = random.choice(nobdr_sigs)
            start = random.randint(0, len(y) - wav_length - 1)
            stop  = start + wav_length
            sig = y[start:stop]
            target = torch.zeros((label_length, 1))
        elif random.random() < hparams["concat_ratio"]:
            target = torch.zeros((label_length, 1))
            boundary = random.randint(label_length // 4, label_length // 4 * 3)
            wav_real = random.choice(nobdr_sigs)
            wav_fake = random.choice(nobdr_sigs)
            sig = torch.zeros_like(wav_real[:wav_length])
            boundary_idx = int(boundary * hparams["time_resolution"] * hparams["sample_rate"])
            sec_len = wav_length - boundary_idx

            real_s = random.randint(0, len(wav_real) - boundary_idx - 1)
            sig[:boundary_idx] = wav_real[real_s:real_s+boundary_idx]

            fake_s = random.randint(0, len(wav_fake) - sec_len - 1)
            sig[boundary_idx:] = wav_fake[fake_s:fake_s+sec_len]
            lab_start = boundary - hparams["neighborhood_size"]
            lab_end = boundary + hparams["neighborhood_size"]
            if lab_start < 0:
                lab_start = 0
            if lab_end > label_length:
                lab_end = label_length
            target[lab_start:lab_end] = 1

        else:
            y = wav2sig[wav]
            if len(y) - wav_length < 0:
                print(y.shape, wav)

            start = random.randint(0, len(y) - wav_length - 1)
            stop  = start + wav_length
            sig = y[start:stop]

            change_points = [start for start, stop in segs[1:]]
            target = torch.zeros((label_length, 1))
            
            for cp in change_points:
                if start < cp < stop:
                    idx = round((cp - start) / hparams["sample_rate"] / hparams["time_resolution"])
                    lab_start = idx - hparams["neighborhood_size"]
                    if lab_start < 0:
                        lab_start = 0
                    lab_stop  = idx + hparams["neighborhood_size"]
                    if lab_stop > label_length:
                        lab_stop = label_length
                    target[lab_start:lab_stop] = 1

        return sig.float(), target.float()

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "target"])

    return train_data

def valid_dataio_prep_bdr(hparams):
    sample_rate = int(hparams["sample_rate"])
    dev_wav2info = sorted([line.strip().split() for line in open(hparams["add_dev"])])
    eval_wav2sig = {}
    eval_dict = {}

    for i in tqdm.tqdm(range(len(dev_wav2info)), ncols=64):
        wav, info, blable = dev_wav2info[i][0], dev_wav2info[i][1], dev_wav2info[i][2]
        y = sb.dataio.dataio.read_audio(hparams["dev_wav_folder"]+"/wav/"+wav+'.wav')
        eval_wav2sig[wav] = y
        real_audio_len = y.shape[0]
        utt = wav# remove .wav
        info = info.split('/') # each element is start-end-label
        labels = [1 if l.split('-')[-1] == 'T' else 0 for l in info]
        segs = [[int(float(j.split('-')[0]) * sample_rate), int(float(j.split('-')[1]) * sample_rate)] for j in info]
        segs[-1][-1] = real_audio_len

        blabel = 1 if len(labels) > 1 else 0

        assert segs[-1][0] < segs[-1][1]

        eval_dict[utt] = {
            "wav": wav,
            "segs": segs,
            "labels": labels,
            "blabel": blabel
        }

    print(len(eval_dict))
    valid_data = sb.dataio.dataset.DynamicItemDataset(eval_dict)
    datasets = [valid_data]

    sample_rate = int(hparams["sample_rate"])
    wav_length = round(hparams["wav_length"] * hparams["sample_rate"])
    wav_shift= round(hparams["wav_shift"] * hparams["sample_rate"])
    label_shift = round(hparams["wav_shift"] / hparams["time_resolution"])
    label_length = round(hparams["wav_length"] / hparams["time_resolution"])
    if 'wav2vec' in hparams["feature"] or 'wavLM' in hparams["feature"]:
        wav_length = int((hparams["wav_length"]+hparams["time_resolution"]) * hparams["sample_rate"])
        label_length = int(hparams["wav_length"] / hparams["time_resolution"]) + 1
    @sb.utils.data_pipeline.takes("wav", "segs", "labels", "blabel")
    @sb.utils.data_pipeline.provides("sig", "target", "b_target", "lab_starts", "lab_stops")
    def audio_pipeline(wav, segs, labels, blabel):
        y = eval_wav2sig[wav]
        
        total_label_length = int(y.shape[0] / hparams["sample_rate"] / hparams["time_resolution"])
        target = torch.zeros((total_label_length))

        if len(labels) > 1:
            change_points = [start for start, stop in segs[1:]]
            for cp in change_points:
                idx = round(cp / hparams["sample_rate"] / hparams["time_resolution"])
                lab_start = idx - hparams["neighborhood_size"]
                if lab_start < 0:
                    lab_start = 0
                lab_stop  = idx + hparams["neighborhood_size"]
                target[lab_start:lab_stop] = 1

        wavs, lab_starts, lab_stops = [], [], []
        for i in range(0, max(1, round(y.shape[0] / wav_shift))):
            ws = i * wav_shift
            we = ws + wav_length
            ls = i * label_shift
            le = ls + label_length
            if we > y.shape[0]:
                if y.shape[0] < wav_length:
                    y = torch.cat((y, torch.zeros((wav_length-y.shape[0]))))
                    assert y.shape[0] == wav_length
                if we - ws < wav_length:
                    y = torch.cat(y, torch.zeros((wav_length-y.shape[0])))
                we = y.shape[0]
                ws = we - wav_length
                le = round(y.shape[0] / hparams["sample_rate"] / hparams["time_resolution"])
                ls = le - label_length
                if ws < 0:
                    ws = 0
                    ls = 0
                wavs.append(y[ws:we])
                lab_starts.append(ls)
                lab_stops.append(le)
                break
            wavs.append(y[ws:we])
            lab_starts.append(ls)
            lab_stops.append(le)

        return torch.stack(wavs).float(), target.long(), torch.tensor([blabel]), lab_starts, lab_stops, wav

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set outputs
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "target", "b_target", "lab_starts", "lab_stops", "wav"])

    return valid_data

def valid_dataio_addtest_bdr(hparams):
    sample_rate = int(hparams["sample_rate"])
    test_wav2info = sorted([line.strip() for line in open(hparams["add_test"])])
    eval_wav2sig = {}
    eval_dict = {}

    for i in tqdm.tqdm(range(len(test_wav2info)), ncols=64):
        wav = test_wav2info[i].replace('.wav', '')
        y = sb.dataio.dataio.read_audio(hparams["test_wav_folder"]+"/wav/"+wav+'.wav')
        eval_wav2sig[wav] = y
        real_audio_len = y.shape[0]
        utt = wav# remove .wav
    
        eval_dict[utt] = {
            "wav": wav,
            "blabel": 1
        }

    print(len(eval_dict))
    valid_data = sb.dataio.dataset.DynamicItemDataset(eval_dict)
    datasets = [valid_data]

    sample_rate = int(hparams["sample_rate"])

    wav_length = round(hparams["wav_length"] * hparams["sample_rate"])
    wav_shift= round(hparams["wav_shift"] * hparams["sample_rate"])
    label_shift = round(hparams["wav_shift"] / hparams["time_resolution"])
    label_length = round(hparams["wav_length"] / hparams["time_resolution"])

    if 'wav2vec' in hparams["feature"] or 'wavLM' in hparams["feature"]:
        wav_length = int((hparams["wav_length"]+hparams["time_resolution"]) * hparams["sample_rate"])
        label_length = int(hparams["wav_length"] / hparams["time_resolution"]) + 1

    @sb.utils.data_pipeline.takes("wav", "blabel")
    @sb.utils.data_pipeline.provides("sig", "target", "b_target", "lab_starts", "lab_stops")
    def audio_pipeline(wav, blabel):
        y = eval_wav2sig[wav]
        
        total_label_length = int(y.shape[0] / hparams["sample_rate"] / hparams["time_resolution"])
        target = torch.zeros((total_label_length))

        wavs, lab_starts, lab_stops = [], [], []
        for i in range(0, max(1, round(y.shape[0] / wav_shift))):
            ws = i * wav_shift
            we = ws + wav_length
            ls = i * label_shift
            le = ls + label_length
            if we > y.shape[0]:
                if y.shape[0] < wav_length:
                    y = torch.cat((y, torch.zeros((wav_length-y.shape[0]))))
                    assert y.shape[0] == wav_length
                if we - ws < wav_length:
                    y = torch.cat(y, torch.zeros((wav_length-y.shape[0])))
                we = y.shape[0]
                ws = we - wav_length
                le = round(y.shape[0] / hparams["sample_rate"] / hparams["time_resolution"])
                ls = le - label_length
                if ws < 0:
                    ws = 0
                    ls = 0
                wavs.append(y[ws:we])
                lab_starts.append(ls)
                lab_stops.append(le)
                break
            wavs.append(y[ws:we])
            lab_starts.append(ls)
            lab_stops.append(le)

        return torch.stack(wavs).float(), target.long(), torch.tensor([blabel]), lab_starts, lab_stops, wav

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set outputs
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "target", "b_target", "lab_starts", "lab_stops", "wav"])

    return valid_data

def valid_dataio_prep_bdr_wfpath(hparams, test_info_file, test_wav_dir):
    sample_rate = int(hparams["sample_rate"])
    dev_wav2info = sorted([line.strip().split() for line in open(test_info_file)])
    eval_wav2sig = {}
    eval_dict = {}

    for i in tqdm.tqdm(range(len(dev_wav2info)), ncols=64):
        wav, info, blable = dev_wav2info[i][0], dev_wav2info[i][1], dev_wav2info[i][2]
        y = sb.dataio.dataio.read_audio(test_wav_dir+wav+'.wav')
        eval_wav2sig[wav] = y
        real_audio_len = y.shape[0]
        utt = wav# remove .wav
        info = info.split('/') # each element is start-end-label
        labels = [1 if l.split('-')[-1] == 'T' else 0 for l in info]
        segs = [[int(float(j.split('-')[0]) * sample_rate), int(float(j.split('-')[1]) * sample_rate)] for j in info]
        segs[-1][-1] = real_audio_len

        blabel = 1 if len(labels) > 1 else 0

        assert segs[-1][0] < segs[-1][1]

        eval_dict[utt] = {
            "wav": wav,
            "segs": segs,
            "labels": labels,
            "blabel": blabel
        }

    print(len(eval_dict))
    valid_data = sb.dataio.dataset.DynamicItemDataset(eval_dict)
    datasets = [valid_data]

    sample_rate = int(hparams["sample_rate"])
    wav_length = round(hparams["wav_length"] * hparams["sample_rate"])
    wav_shift= round(hparams["wav_shift"] * hparams["sample_rate"])
    label_shift = round(hparams["wav_shift"] / hparams["time_resolution"])
    label_length = round(hparams["wav_length"] / hparams["time_resolution"])
    if 'wav2vec' in hparams["feature"] or 'wavLM' in hparams["feature"]:
        wav_length = int((hparams["wav_length"]+hparams["time_resolution"]) * hparams["sample_rate"])
        label_length = int(hparams["wav_length"] / hparams["time_resolution"]) + 1
    @sb.utils.data_pipeline.takes("wav", "segs", "labels", "blabel")
    @sb.utils.data_pipeline.provides("sig", "target", "b_target", "lab_starts", "lab_stops")
    def audio_pipeline(wav, segs, labels, blabel):
        y = eval_wav2sig[wav]
        
        total_label_length = int(y.shape[0] / hparams["sample_rate"] / hparams["time_resolution"])
        target = torch.zeros((total_label_length))

        if len(labels) > 1:
            change_points = [start for start, stop in segs[1:]]
            for cp in change_points:
                idx = round(cp / hparams["sample_rate"] / hparams["time_resolution"])
                lab_start = idx - hparams["neighborhood_size"]
                if lab_start < 0:
                    lab_start = 0
                lab_stop  = idx + hparams["neighborhood_size"]
                target[lab_start:lab_stop] = 1

        wavs, lab_starts, lab_stops = [], [], []
        for i in range(0, max(1, round(y.shape[0] / wav_shift))):
            ws = i * wav_shift
            we = ws + wav_length
            ls = i * label_shift
            le = ls + label_length
            if we > y.shape[0]:
                if y.shape[0] < wav_length:
                    y = torch.cat((y, torch.zeros((wav_length-y.shape[0]))))
                    assert y.shape[0] == wav_length
                if we - ws < wav_length:
                    y = torch.cat(y, torch.zeros((wav_length-y.shape[0])))
                we = y.shape[0]
                ws = we - wav_length
                le = round(y.shape[0] / hparams["sample_rate"] / hparams["time_resolution"])
                ls = le - label_length
                if ws < 0:
                    ws = 0
                    ls = 0
                wavs.append(y[ws:we])
                lab_starts.append(ls)
                lab_stops.append(le)
                break
            wavs.append(y[ws:we])
            lab_starts.append(ls)
            lab_stops.append(le)

        return torch.stack(wavs).float(), target.long(), torch.tensor([blabel]), lab_starts, lab_stops, wav

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set outputs
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "target", "b_target", "lab_starts", "lab_stops", "wav"])

    return valid_data


def valid_dataio_prep_bdr_wfpath_add2022(hparams, test_info_file, test_wav_dir):
    sample_rate = int(hparams["sample_rate"])
    dev_wav2info = sorted([line.strip().split()[0] for line in open(test_info_file)])
    eval_wav2sig = {}
    eval_dict = {}

    for i in tqdm.tqdm(range(len(dev_wav2info)), ncols=64):
        wav = dev_wav2info[i]
        y = sb.dataio.dataio.read_audio(test_wav_dir+wav+'.wav')
        eval_wav2sig[wav] = y
        real_audio_len = y.shape[0]
        utt = wav
        
        eval_dict[utt] = {
            "wav": wav,
            "blabel": 1
        }

    print(len(eval_dict))
    valid_data = sb.dataio.dataset.DynamicItemDataset(eval_dict)
    datasets = [valid_data]

    sample_rate = int(hparams["sample_rate"])
    wav_length = round(hparams["wav_length"] * hparams["sample_rate"])
    wav_shift= round(hparams["wav_shift"] * hparams["sample_rate"])
    label_shift = round(hparams["wav_shift"] / hparams["time_resolution"])
    label_length = round(hparams["wav_length"] / hparams["time_resolution"])

    if 'wav2vec' in hparams["feature"] or 'wavLM' in hparams["feature"]:
        wav_length = int((hparams["wav_length"]+hparams["time_resolution"]) * hparams["sample_rate"])
        label_length = int(hparams["wav_length"] / hparams["time_resolution"]) + 1

    @sb.utils.data_pipeline.takes("wav", "blabel")
    @sb.utils.data_pipeline.provides("sig", "target", "b_target", "lab_starts", "lab_stops")
    def audio_pipeline(wav, blabel):
        y = eval_wav2sig[wav]
        
        total_label_length = int(y.shape[0] / hparams["sample_rate"] / hparams["time_resolution"])
        target = torch.zeros((total_label_length))

        wavs, lab_starts, lab_stops = [], [], []
        for i in range(0, max(1, round(y.shape[0] / wav_shift))):
            ws = i * wav_shift
            we = ws + wav_length
            ls = i * label_shift
            le = ls + label_length
            if we > y.shape[0]:
                if y.shape[0] < wav_length:
                    y = torch.cat((y, torch.zeros((wav_length-y.shape[0]))))
                    assert y.shape[0] == wav_length
                if we - ws < wav_length:
                    y = torch.cat(y, torch.zeros((wav_length-y.shape[0])))
                we = y.shape[0]
                ws = we - wav_length
                le = round(y.shape[0] / hparams["sample_rate"] / hparams["time_resolution"])
                ls = le - label_length
                if ws < 0:
                    ws = 0
                    ls = 0
                wavs.append(y[ws:we])
                lab_starts.append(ls)
                lab_stops.append(le)
                break
            wavs.append(y[ws:we])
            lab_starts.append(ls)
            lab_stops.append(le)

        return torch.stack(wavs).float(), target.long(), torch.tensor([blabel]), lab_starts, lab_stops, wav

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set outputs
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "target", "b_target", "lab_starts", "lab_stops", "wav"])

    return valid_data
