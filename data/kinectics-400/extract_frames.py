import argparse
import subprocess
import os
import re
import pickle as pkl
from multiprocessing import Event, Process, Queue
import queue
import shlex


def call(event, q, oq):
    while not event.is_set():
        try:
            cmd, vid = q.get(timeout=1)
        except queue.Empty:
            continue
        try:
            output = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
            oq.put((output, vid))
        except subprocess.CalledProcessError as e:
            oq.put((-1, vid))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='/raw')
    parser.add_argument('--output', default='processed')
    args = parser.parse_args()
    data_root = args.video
    output_root = args.output
    s = 0
    event = Event()
    q = Queue(20)
    oq = Queue(100)
    pool = [Process(target=call, args=(event, q, oq)) for i in range(4)]
    for p in pool:
        p.start()
    for subset in ('valid', 'train', ):
        if os.path.exists('frames-{}.txt'.format(subset)):
            with open('frames-{}.txt'.format(subset), 'r') as f:
                processed = set([line.strip().split()[0] for line in f])
        else:
            processed = dict()
        with open('frames-{}.txt'.format(subset), 'a') as fo:
            subset_root = os.path.join(data_root, subset)
            for class_name in os.listdir(subset_root):
                class_root = os.path.join(subset_root, class_name)
                print(class_name, len(os.listdir(class_root)))
                s += len(os.listdir(class_root))
                for video_file in os.listdir(class_root):
                    vid, ext = os.path.splitext(video_file)
                    if ext != '.mp4' or vid in processed:
                        continue
                    output_path = os.path.join(output_root, subset, class_name, vid)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    cmd = "ffmpeg -i {} -r 25 -s 320x240 {}".format(os.path.join(class_root, video_file), os.path.join(output_path, '%06d.jpg'))
                    q.put((cmd, vid))
                    while True:
                        try:
                            output, vid = oq.get_nowait()
                        except queue.Empty:
                            break
                        if output == -1:
                            print('Failed', vid)
                            continue
                        output_info = output.decode().split('\n')[-3]
                        frames = re.match(r'^frame=\s*(\d+)', output_info).group(1)
                        print('{}\t{}'.format(output_path, frames))
                        fo.write('{} {}\n'.format(vid, frames))
        print('{} videos processed'.format(s))
    q.close()
    event.set()
    for p in pool:
        p.join()
