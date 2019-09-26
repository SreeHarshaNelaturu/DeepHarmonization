import caffe
import numpy as np
from PIL import Image
import os
import runway
#import tensorflow as tf
"""
# result folder
#folder_name = 'result/'
#if os.path.isdir(folder_name):
#  pass
#else:
#  os.makedirs(folder_name)

# load test image list
#filename = 'data/list.txt'
#with open(filename, 'r') as f:
#  path_src = [line.rstrip() for line in f.readlines()]

# set up caffe
#caffe.set_device(0)
#caffe.set_mode_gpu()
"""
#if tf.test.is_gpu_available():
#    print("Using GPU")
#    caffe.set_mode_gpu()
#else:
#    print("Using CPU")
caffe.set_mode_cpu()
# load net

@runway.setup(options={'prototxt': runway.file(extension='.prototxt'),'caffemodel' : runway.file(extension='.caffemodel')})
def setup(opts):
    net = caffe.Net(opts['prototxt'], opts['caffemodel'], caffe.TEST)
    return net

inputs = {"input_image" : runway.image, "masked_image" : runway.image}
#mask =  {"masked_image" : runway.image}
outputs = {"harmonized_image" : runway.image}

size = np.array([512,512])

@runway.command('Harmonize Image', input=inputs, output=outputs, description="Harmonize Image")
def harmonize_image(net, input):
    im_ori = Image.open(input["input_image"])
    im = im_ori.resize(size, Image.BICUBIC)
    im = np.array(im, dtype=np.float32)
    if im.shape[2] == 4:
        im = im[:,:,0:3]

    im = im[:,:,::-1]
    im -= np.array((104.00699, 116.66877, 122.67892))
    im = im.transpose((2,0,1))

    mask = Image.open(input["masked_image"])
    mask = mask.resize(size, Image.BICUBIC)
    mask = np.array(mask, dtype=np.float32)
    if len(mask.shape) == 3:
        mask = mask[:,:,0]

    mask -= 128.0
    mask = mask[np.newaxis, ...]

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im

    net.blobs['mask'].reshape(1, *mask.shape)
    net.blobs['mask'].data[...] = mask

    # run net for prediction
    net.forward()
    out = net.blobs['output-h'].data[0]
    out = out.transpose((1,2,0))
    out += np.array((104.00699, 116.66877, 122.67892))
    out = out[:,:,::-1]

    neg_idx = out < 0.0
    out[neg_idx] = 0.0
    pos_idx = out > 255.0
    out[pos_idx] = 255.0
    # save result
    result = out.astype(np.uint8)
    result = Image.fromarray(result)

    im = im_ori.resize(size, Image.BICUBIC)
    im = np.array(im, dtype=np.uint8)
    if im.shape[2] == 4:
       im = im[:,:,0:3]

    #end = path_.find('.')
    result_all = np.concatenate((im, result), axis = 1)
    result_all = Image.fromarray(result_all)
    return result_all
