import os
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the SpeakingFaces dataset")
ap.add_argument("-f", "--filename", default="train_list.txt",
                help="the name of the output file")
ap.add_argument("-i", "--sub_range",  nargs='+', type=int, default=(1, 100),
                help="range of subjects")
args = vars(ap.parse_args())

sub_id_str = args["sub_range"][0]
sub_id_end = args["sub_range"][1]
path_to_dataset = args["dataset"]

# for each utterance of each subject, add a line to the file following this patter
# sub_1 sub_1/wav/1_1_2_7_217_1 
result = []
for sub_id in range(sub_id_str, sub_id_end + 1):
    key = "sub_{}".format(sub_id)
    value_base = os.path.join(key, "wav")

    path_files = os.path.join(path_to_dataset, value_base, "*")
    list_filepaths = glob.glob(path_files)
    for filepath in list_filepaths:
        filename = filepath.split('/')[-1]
        value = os.path.join(value_base, filename)
        result.append((key, value))

with open(args["filename"],"w") as fp:
    fp.write('\n'.join('{} {}'.format(x[0], x[1]) for x in result))
