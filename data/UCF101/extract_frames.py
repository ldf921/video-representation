import subprocess
import os
import re

data_root = 'raw'
output_root = 'processed'
with open('frames.txt', 'w') as fo:
    for video_file in os.listdir(data_root):
        vid, _ = os.path.splitext(video_file)
        output_path = os.path.join(output_root, vid)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output = subprocess.check_output("ffmpeg -i {} -r 25 -s 320x240 {}".format(os.path.join(data_root, video_file), os.path.join(output_path, '%06d.jpg')),
               stderr=subprocess.STDOUT, shell=True)
        output_info = output.decode().split('\n')[-3]
        print('{}\t{}'.format(output_path, output_info))
        frames = re.match(r'^frame=\s*(\d+)', output_info).group(1)
        print(frames)
        fo.write('{} {}\n'.format(vid, frames))
