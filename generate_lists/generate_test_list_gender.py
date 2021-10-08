import pandas as pd
import os
from imutils import paths
import glob
import numpy as np
import itertools
import random
import argparse
import copy

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
ap.add_argument("-m", "--metadata",  required = True,
                help="path to the SpeakingFaces metadata")


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
        if num_pos[key] > (n*(n-1)/2):
            result = False
    return result

path_to_data = args["dataset"]
num_neg = args["neg"]
filename = args["filename"]
sub_id_str = args["sub_range"][0]
sub_id_end = args["sub_range"][1]

path_subjects = args["metadata"]
df_subjects = pd.read_csv(os.path.join(path_subjects, "subjects.csv")).loc[sub_id_str - 1:sub_id_end - 1]
males = df_subjects[df_subjects.Gender == 'Male'].Sub_ID.values
females = df_subjects[df_subjects.Gender == 'Female'].Sub_ID.values

sub_ids = range(sub_id_str, sub_id_end+1)

utterances = dict()
pos_pairs = dict()
neg_pairs = dict()
num_pos = dict()
other_ids = dict()

for sub_id in sub_ids:
    utterances[sub_id] = get_sub_utterances(sub_id, path_to_data)
    if sub_id in males:
        num_pos[sub_id] = num_neg*(len(males)-1)
        other_ids[sub_id] = list(copy.deepcopy(males))
    else:    
        num_pos[sub_id]  = num_neg*(len(females)-1)
        other_ids[sub_id] = list(copy.deepcopy(females))

if check_num_pos(num_pos, utterances):

    for sub_id in sub_ids:

        all_pos_pairs =  list(itertools.combinations(utterances[sub_id],2))
        pos_pairs[sub_id] = random.sample(all_pos_pairs, num_pos[sub_id])
        other_ids[sub_id].remove(sub_id)
        neg_pairs[sub_id] = []

        for other_id in other_ids[sub_id]:
            all_neg_pairs =  list(itertools.product(utterances[sub_id],utterances[other_id]))
            neg_pairs[sub_id].append(random.sample(all_neg_pairs, num_neg))
        neg_pairs[sub_id] = list(itertools.chain(*neg_pairs[sub_id]))

    all_pairs_labeled = []
    for sub_id in sub_ids:
        for i in range(num_pos[sub_id]):
            all_pairs_labeled.append((1, pos_pairs[sub_id][i][0], pos_pairs[sub_id][i][1]))
            all_pairs_labeled.append((0, neg_pairs[sub_id][i][0], neg_pairs[sub_id][i][1]))

    with open(filename,"w") as fp:
        fp.write('\n'.join('{} {} {}'.format(x[0], x[1], x[2]) for x in all_pairs_labeled))

else:
    print("insufficient number of positive pairs")

