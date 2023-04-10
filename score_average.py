import pandas as pd
import numpy as np
import argparse
import os
from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf, recordPredictions

parser = argparse.ArgumentParser()
parser.add_argument('--path_1', required=True, type=str, help='Path to the 1st predictions')
parser.add_argument('--path_2', required=True, type=str, help='Path to the 2nd predictions')
parser.add_argument('--path_3', default="", type=str, help='Path to the 3d predictions')
parser.add_argument('--path_4', default="", type=str, help='Path to the 4th predictions')
parser.add_argument('--path_5', default="", type=str, help='Path to the 5th predictions')

def compute_scores(mean_scores, true_labels):
    p_target = 0.05
    c_miss = 1
    c_fa = 1

    result = tuneThresholdfromScore(mean_scores, true_labels, [1, 0.1])

    fnrs, fprs, thresholds = ComputeErrorRates(mean_scores, true_labels)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)
    return result


def simple_fusion(scores_1, scores_2, scores_3, scores_4, scores_5):
    data_1 = pd.read_fwf(scores_1, sep=' ', header=None)
    data_2 = pd.read_fwf(scores_2, sep=' ', header=None)
    data_3 = pd.read_fwf(scores_3, sep=' ', header=None)
    data_4 = pd.read_fwf(scores_4, sep=' ', header=None)
    data_5 = pd.read_fwf(scores_5, sep=' ', header=None)
    true_labels = data_1[2]
    
    mean_scores = (data_1[0] + data_2[0] + data_3[0] + data_4[0] + data_5[0]) / 5
    result = compute_scores(mean_scores, true_labels)
    eer = result[1]
    print('wav+rgb+thr+wavrgb+wavrgbthr EER: {}'.format(eer))

    mean_scores = (data_1[0] + data_2[0] + data_3[0] + data_4[0]) / 4
    result = compute_scores(mean_scores, true_labels)
    eer = result[1]
    print('wav+rgb+thr+wavrgb EER: {}'.format(eer))

    mean_scores = (data_1[0] + data_2[0] + data_4[0] + data_5[0]) / 4
    result = compute_scores(mean_scores, true_labels)
    eer = result[1]
    print('wav+rgb+wavrgb+wavrgbthr EER: {}'.format(eer))

  
    mean_scores = (data_1[0] + data_2[0] + data_4[0]) / 3
    result = compute_scores(mean_scores, true_labels)
    eer = result[1]
    print('wav+rgb+wavrgb EER: {}'.format(eer))

    mean_scores = (data_1[0] + data_2[0] + data_3[0]) / 3
    result = compute_scores(mean_scores, true_labels)
    eer = result[1]
    print('wav+rgb+thr EER: {}'.format(eer))

    mean_scores = (data_1[0] + data_2[0]) / 2
    result = compute_scores(mean_scores, true_labels)
    eer = result[1]
    print('wav+rgb EER: {}'.format(eer))


   
args = parser.parse_args()
simple_fusion(args.path_1, args.path_2, args.path_3, args.path_4,  args.path_5)
