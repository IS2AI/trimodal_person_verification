import pandas as pd
import numpy as np
import argparse
import os
from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf, recordPredictions

parser = argparse.ArgumentParser()
parser.add_argument('--path_rgb', required=True, type=str, help='Path to the visual predictions')
parser.add_argument('--path_wav', required=True, type=str, help='Path to the audio predictions')
parser.add_argument('--path_thr', default="", type=str, help='Path to the thermal predictions')
#parser.add_argument('--save_path', required=True, type=str, help='Path for saving the predictions')
#parser.add_argument('--test_list', required=True, type=str, help='Evaluation list for which to save predictions')


def simple_fusion(rgb_scores_filename, thr_scores_filename, wav_scores_filename, save_path, test_list):
    rgb_data = pd.read_fwf(rgb_scores_filename, sep=' ', header=None)
    wav_data = pd.read_fwf(wav_scores_filename, sep=' ', header=None)

    if thr_scores_filename != "":
        thr_data = pd.read_fwf(thr_scores_filename, sep=' ', header=None)
        mean_scores = (rgb_data[0] + thr_data[0] + wav_data[0]) / 3
    else:
        mean_scores = (rgb_data[0] + wav_data[0]) / 2

    true_labels = rgb_data[2]
    p_target = 0.05
    c_miss = 1
    c_fa = 1

    result = tuneThresholdfromScore(mean_scores, true_labels, [1, 0.1])

    fnrs, fprs, thresholds = ComputeErrorRates(mean_scores, true_labels)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

#    if not os.path.exists(save_path):
#        os.makedirs(save_path)
#    recordPredictions(mean_scores, result[4], save_path, test_list, True)

    eer = result[1]
    print('EER: {eer:0.4f}, MinDCF: {mindcf:0.4f}')
#    return eer, mindcf


args = parser.parse_args()
simple_fusion(args.path_rgb, args.path_thr, args.path_wav, args.save_path, args.test_list)
