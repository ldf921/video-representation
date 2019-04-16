import numpy as np
from torch.utils.data import Sampler

class RandomSampler(Sampler):
    def __init__(self, data_source, loop):
        self.data_source = data_source
        self.loop = loop

    def __iter__(self):
        return self.generator()

    def generator(self):
        vlen = 1 + (self.data_source.length - 1) * self.data_source.stride
        for t in range(self.loop):
            for vid in np.random.permutation(len(self.data_source)):
                t = len(self.data_source.videos[vid]) - vlen + 1
                if t > 0:
                    s = np.random.randint(t)
                else:
                    s = 0
                yield (vid, s)

    def __len__(self):
        return len(self.data_source) * self.loop


class ValSampler(Sampler):
    def __init__(self, data_source, stride):
        self.data_source = data_source
        self.clip_indices = []
        for vid in range(len(self.data_source)):
            vlen = stride * self.data_source.stride
            N = len(self.data_source.videos[vid])
            k = (N - 1) // vlen + 1
            M = max(0, N - vlen + 1)
            samples = [(vid, 0),] + [(vid, i * M // (k - 1)) for i in range(1, k)]
            self.clip_indices.extend(samples)

    def __iter__(self):
        return iter(self.clip_indices)

    def __len__(self):
        return len(self.clip_indices)
