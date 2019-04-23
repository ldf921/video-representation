Prepare
============
Install ffmpeg to extract frames from videos. Install PyTorch 1.0. Install tqdm.

Install unrar. Download the datasets and extract frames. The train, val, test split is provided.
```bash
cd ./data/UCF101
download.sh
python3 extract_frames.py
```

Set your environment variables in `path.py`.
```python3
REPO_ROOT = '{repo location}/video-representation' # root of the repo
UCF101_ROOT = ./data/UCF101 # root for UCF-101 data set
KINETICS_ROOT = '/data/kinetics-400'
```


Train
============
Usage 
```bash
python3 train.py config/convlstm -b 8 -p 200 -t 4
```

