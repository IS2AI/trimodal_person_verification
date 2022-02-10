#!/usr/bin/pythonl
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import test_dataset_loader

from torch.cuda.amp import autocast, GradScaler

class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None, eval_mode=False):
        return self.module(x, label, eval_mode)


class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__();

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs);
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs);

        self.nPerSpeaker = nPerSpeaker
        self.modality = kwargs["modality"].lower()
        self.eval = kwargs["eval"]

    def forward(self, data, label=None, eval_mode=False):

        if not eval_mode:
            if "wav" in self.modality and "rgb" in self.modality and "thr" in self.modality:
                data_wav = data[0]
                old_shape_wav = list(data_wav.shape)
                new_shape_wav = [old_shape_wav[0] * old_shape_wav[1]] + old_shape_wav[2:]
                data_wav = data_wav.reshape(new_shape_wav).cuda()

                data_rgb = data[1]
                old_shape_rgb = list(data_rgb.shape)
                new_shape_rgb = [old_shape_rgb[0] * old_shape_rgb[1]] + old_shape_rgb[2:]
                data_rgb = data_rgb.reshape(new_shape_rgb).cuda()

                data_thr = data[2]
                old_shape_thr = list(data_thr.shape)
                new_shape_thr = [old_shape_thr[0] * old_shape_thr[1]] + old_shape_thr[2:]
                data_thr = data_thr.reshape(new_shape_thr).cuda()

                data = data_wav, data_rgb, data_thr

            elif "wav" in self.modality and "rgb" in self.modality:
                data_wav = data[0]
                old_shape_wav = list(data_wav.shape)
                new_shape_wav = [old_shape_wav[0] * old_shape_wav[1]] + old_shape_wav[2:]
                data_wav = data_wav.reshape(new_shape_wav).cuda()

                data_rgb = data[1]
                old_shape_rgb = list(data_rgb.shape)
                new_shape_rgb = [old_shape_rgb[0] * old_shape_rgb[1]] + old_shape_rgb[2:]
                data_rgb = data_rgb.reshape(new_shape_rgb).cuda()

                data = data_wav, data_rgb

            elif "wav" in self.modality and "thr" in self.modality:
                data_wav = data[0]
                old_shape_wav = list(data_wav.shape)
                new_shape_wav = [old_shape_wav[0] * old_shape_wav[1]] + old_shape_wav[2:]
                data_wav = data_wav.reshape(new_shape_wav).cuda()

                data_thr = data[1]
                old_shape_thr = list(data_thr.shape)
                new_shape_thr = [old_shape_thr[0] * old_shape_thr[1]] + old_shape_thr[2:]
                data_thr = data_thr.reshape(new_shape_thr).cuda()

                data = data_wav, data_thr
            else:
                old_shape = list(data.shape)
                new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
                data = data.reshape(new_shape).cuda()

        outp = self.__S__.forward(data, eval_mode)

        if label == None:
            return outp

        else:
            outp = outp.reshape(-1, self.nPerSpeaker, outp.size()[-1]).squeeze(1)
            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1


class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__  = speaker_model
        self.modality = kwargs["modality"].lower()
        self.num_images = kwargs["num_images"]

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler() 

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0;    # EER or accuracy

        tstart = time.time()
        
        for data, data_label in loader:
            self.__model__.zero_grad();

            label   = torch.LongTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label, False)
                self.scaler.scale(nloss).backward();
                self.scaler.step(self.__optimizer__);
                self.scaler.update();       
            else:
                nloss, prec1 = self.__model__(data, label, False)
                nloss.backward();
                self.__optimizer__.step();


            loss    += nloss.detach().cpu().item();
            top1    += prec1.detach().cpu().item();
            counter += 1;
            index   += stepsize;

        

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing ({:d}) ".format(index));
                sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss/counter, top1/counter, stepsize/telapsed));
                sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()
        
        return (loss/counter, top1/counter);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, print_interval=100, **kwargs):
        
        self.__model__.eval();
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = sum([x.strip().split()[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()
        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )

        for idx, data in enumerate(test_loader):
            if "wav" in self.modality and "rgb" in self.modality and "thr" in self.modality:
                inp1 = data[0][0][0].cuda(), data[0][1][0].cuda(), data[0][2][0].cuda()

            elif "wav" in self.modality and "rgb" in self.modality:
                inp1 = data[0][0][0].cuda(), data[0][1][0].cuda()

            elif "wav" in self.modality and "thr" in self.modality:
                inp1 = data[0][0][0].cuda(), data[0][1][0].cuda()

            else:
                inp1                = data[0][0].cuda()
            ref_feat            = self.__model__(inp1, None, True).detach().cpu()
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__model__.module.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();

            score = -1 * numpy.mean(dist);

            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        return (all_scores, all_labels, all_trials);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.__model__.module.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict();
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("{} is not in the model.".format(origname));
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

