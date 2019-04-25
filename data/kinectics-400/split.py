import re
import os
import json

import numpy as np
import pickle as pkl


if __name__ == '__main__':
    with open('resources/classes.json', 'r') as fi:
        classes = {class_name.replace(' ', '_') : i for i, class_name in enumerate(json.load(fi))}
    data_root = 'processed'
    s = 0
    for subset in ('valid', ):
        frames = []
        labels = []
        with open('frames-{}.txt'.format(subset), 'r') as f:
            processed = dict([line.strip().split() for line in f])
        subset_root = os.path.join(data_root, subset)
        for class_name in os.listdir(subset_root):
            class_root = os.path.join(subset_root, class_name)
            print(class_name, len(os.listdir(class_root)))
            s += len(os.listdir(class_root))
            for vid in os.listdir(class_root):
                if vid in processed:
                    frames.append((os.path.join(class_root, vid), int(processed[vid])))
                    labels.append(classes[class_name])
        with open('{}.pkl'.format(subset), 'wb') as fo:
            pkl.dump(dict(frames=frames, labels=labels), fo)
    print('Total {} videos'.format(s))

