import os
import pickle as pk

def parse_sequence_file(path_to_file):
    with open(os.path.join(path_to_file), 'r') as f:
        lines = f.readlines()
    sequences = [line.strip() for line in lines if (not (line[0].isdigit() or line == ''))]
    print(sequences)
    dict = {}
    key = 0
    for sequence in sequences:
        if sequence not in dict.keys():
            dict[sequence] = key
            key+=1
    print(dict)
    with open('seq2idmap.pk', 'wb') as p:
        pk.dump(dict, p)

path_to_file =  'D:/Users Data/ido.iGIP1/Downloads/scripts/subjects_and_sequences.txt'

parse_sequence_file(path_to_file)

with open('seq2idmap.pk', 'rb') as r:
    dict_read = pk.load(r)

print(dict_read)