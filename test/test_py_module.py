import os, sys
from pathlib import Path
import skvideo.datasets, skvideo.io
import skimage
import skimage.transform, skimage.io
import matplotlib.pyplot as plt
import time

script_path = Path(os.path.dirname(__file__))
project_path = script_path.parent
bin_path = project_path/'bin'

if str(bin_path) not in sys.path:
    sys.path.append(str(bin_path))

import json
import py_vidscramble


# define the pipeline
pipeline_spec = {
    'data_embed_block_size': 8,
    'data_embed_num_rows': 4,
    'data_embed_interval': 60,
    'steps' : [
        {
            'name' : 'ImageShift',
            'sx': 1,
            'sy': -1
        },
        {
            'name' : 'RowShuffle',
            'row_group_size': 8,
            'random_seed': 42
        },
        {
            'name' : 'ImageTranspose'
        },
        {
            'name' : 'RowShuffle',
            'row_group_size': 8,
            'random_seed': 300
        },
        {
            'name' : 'ImageTranspose'
        },
        {
            'name' : 'ImageShift',
            'sx': -1,
            'sy': 1
        },
        # {
        #     'name' : 'RowMix',
        #     'row_group_size': 32,
        #     'random_seed': 89
        # },
    ]
}

pipeline_spec_json = json.dumps(pipeline_spec)
pipeline = py_vidscramble.build_pipeline_from_json(pipeline_spec_json)

#video_frames = skvideo.io.vread(skvideo.datasets.bigbuckbunny())
video_frames = skvideo.io.vread('r6.mp4')
pipeline.fit(video_frames[0])

print(pipeline.to_json())
# embed_json = pipeline.to_json_image()
# plt.imshow(embed_json)
# plt.show()

def test_frame_time():
    st = time.time()
    for frame in video_frames:
        new_img = pipeline.transform(frame)
    et = time.time()
    print('time per frame:', (et-st)/len(frame) * 1000)

def test_video_forward():
    new_frames = []
    for frame in video_frames:
        new_img = pipeline.transform(frame)
        new_frames.append(new_img)

    skvideo.io.vwrite('test.mp4', new_frames)


def test_forward():
    new_img = pipeline.transform(video_frames[0])
    plt.imshow(new_img)
    plt.show()

def test_forward_backward():
    new_img = pipeline.transform(video_frames[0])
    new_img_inv = pipeline.inverse_transform(new_img)
    # plt.subplot(131)
    # plt.imshow(video_frames[0])
    plt.subplot(121)
    plt.imshow(new_img)
    plt.subplot(122)
    plt.imshow(new_img_inv)
    plt.show()

def test_info_recovery():
    new_img = pipeline.transform(video_frames[0])
    new_img = skimage.transform.rescale(new_img, (1.0,1.2), channel_axis=2)
    new_img = skimage.img_as_ubyte(new_img)
    skimage.io.imsave('test.jpg', new_img)

    # info = py_vidscramble.ImageRecoveryInfo()
    # py_vidscramble.VideoScramblePipeline.get_data_extraction_transform(new_img, info)

    # for key in info.__dir__():
    #     if not key.startswith('_'):
    #         print(key, getattr(info, key))

# test_frame_time()
# test_forward()
# test_forward_backward()
test_video_forward()
# test_info_recovery()