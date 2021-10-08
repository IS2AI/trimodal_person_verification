import pandas as pd
import os
from imutils import paths
import glob
import numpy as np
import itertools
import random
import argparse

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the SpeakingFaces dataset")
ap.add_argument("-f", "--filename", required=True,
                help="the name of the output file")
ap.add_argument("-n", "--neg", type=int, default=10,
                help="the number of negative samples between a pair of subjects")
ap.add_argument("-i", "--sub_range",  nargs='+', type=int, required = True,
                help="range of subjects")
args = vars(ap.parse_args())

def get_sub_utterances(sub_id, path_to_data):
    key = "sub_{}".format(sub_id)
    value_base = os.path.join(key, "wav")
    path_files = os.path.join(path_to_data, value_base, "*")
    list_filepaths = glob.glob(path_files)
    result = [os.path.join(value_base, filepath.split('/')[-1]) for filepath in list_filepaths]
    return result

def check_num_pos(num_pos, utterances):
    result = True
    for key in utterances:
        n = len(utterances[key])
        if num_pos > (n*(n-1)/2):
            result = False
    return result

path_to_data = args["dataset"]
num_neg = args["neg"]
filename = args["filename"]
sub_id_str = args["sub_range"][0]
sub_id_end = args["sub_range"][1]

sub_ids = range(sub_id_str, sub_id_end+1)
num_pos = num_neg*(len(sub_ids)-1)
utterances = dict()

for sub_id in sub_ids:
    utterances[sub_id] = get_sub_utterances(sub_id, path_to_data)

if check_num_pos(num_pos, utterances):
    pos_pairs = dict()
    neg_pairs = dict()


    for sub_id in sub_ids:
        all_pos_pairs =  list(itertools.combinations(utterances[sub_id],2))
        pos_pairs[sub_id] = random.sample(all_pos_pairs, num_pos)

        other_ids = list(sub_ids)
        other_ids.remove(sub_id)
        neg_pairs[sub_id] = []
        for other_id in other_ids:
            all_neg_pairs =  list(itertools.product(utterances[sub_id],utterances[other_id]))
            neg_pairs[sub_id].append(random.sample(all_neg_pairs, num_neg))
        neg_pairs[sub_id] = list(itertools.chain(*neg_pairs[sub_id]))

    all_pairs_labeled = []
    for sub_id in sub_ids:
        for i in range(num_pos):
            all_pairs_labeled.append((1, pos_pairs[sub_id][i][0], pos_pairs[sub_id][i][1]))
            all_pairs_labeled.append((0, neg_pairs[sub_id][i][0], neg_pairs[sub_id][i][1]))

    with open(filename,"w") as fp:
        fp.write('\n'.join('{} {} {}'.format(x[0], x[1], x[2]) for x in all_pairs_labeled))
else:
    print("insufficient number of positive pairs {}".format(num_pos))
