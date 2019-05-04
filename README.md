# Unsupervised Video Representation Learning
Tingfung Lau, Jiarui Lu, Jing Mao

Course project for CMU 10-707 Topics in machine learning. 

## Prepare
The envirnment is Python 3. Install ffmpeg to extract frames from videos. Install PyTorch 1.0. Install tqdm.

Install unrar. Download the datasets and extract frames. The train, val, test split is provided.
```bash
cd ./data/UCF101
bash download.sh
python3 extract_frames.py
```

The kinetics-400 data is downloaded using [kinetics-downloader](https://github.com/Showmax/kinetics-downloader.git), which may takes a few days to download the first 10K videos we used.

Set your environment variables in `path.py`.
```python3
REPO_ROOT = '{repo location}/video-representation' # root of the repo
UCF101_ROOT = data/UCF101 # root for UCF-101 data set
KINETICS_ROOT = data/kinetics-400
```

## Train
Usage 
```bash
python3 train.py config/convlstm -b 8 -p 200 -t 4
```

## Test
Evaluation using test1 split in UCF101.
```bash
python3 train.py checkpoints/convlstm/config --checkpoint 20.model --test test1 -b 32 -p 200 -t 4
```

## Acknowledgements
The sync batchnorm implementation in PyTorch is an open source package by [Jiayuan Mao](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) realeased under MIT license.
