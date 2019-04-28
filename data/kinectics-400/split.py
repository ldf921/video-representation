import argparse
import re
import os
import json

import numpy as np
import pickle as pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=lambda t : t.split(','), default=['train', 'valid'])
    args = parser.parse_args()

    with open('resources/classes.json', 'r') as fi:
        classes = {class_name.replace(' ', '_') : i for i, class_name in enumerate(json.load(fi))}
    data_root = 'processed'
    s = 0
    np.random.seed(233)
    for subset in args.subset:
        frames = []
        labels = []
        with open('frames-{}.txt'.format(subset), 'r') as f:
            processed = dict([line.strip().split() for line in f])
        subset_root = os.path.join(data_root, subset)
        for class_name in os.listdir(subset_root):
            class_root = os.path.join(subset_root, class_name)
            s += len(os.listdir(class_root))
            sum_frames = 0
            nvids = 0
            for vid in os.listdir(class_root):
                if vid in processed:
                    video_root = os.path.join(class_root, vid)
                    imgs = os.listdir(video_root)
                    actual_frames = len(imgs)
                    if max(imgs) == '%06d.jpg' % actual_frames:
                        frames.append((video_root, actual_frames))
                        sum_frames += actual_frames
                        nvids += 1
                        labels.append(classes[class_name])
            if nvids > 0:
                print(class_name, nvids, sum_frames / nvids)
        if True:
            with open('{}.pkl'.format(subset), 'wb') as fo:
                pkl.dump(dict(frames=frames, labels=labels), fo)
        else:
            n = len(frames)
            num_val = int(0.1 * n + 0.5)
            indices = np.random.permutation(n)
            subset_indices = {
                    'train' : indices[num_val:],
                    'val' : indices[:num_val]
                    }
            for subset, indices in subset_indices.items():
                sframes = [ frames[i] for i in sorted(indices) ]
                slabels = [ labels[i] for i in sorted(indices) ]
                with open('{}.pkl'.format(subset), 'wb') as fo:
                    pkl.dump(dict(frames=sframes, labels=slabels), fo)
                print('{} {}'.format(subset, len(sframes)))
            print(n)
    print('Total {} videos'.format(s))

