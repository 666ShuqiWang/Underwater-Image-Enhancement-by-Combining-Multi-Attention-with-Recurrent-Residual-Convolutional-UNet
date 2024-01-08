from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
import scipy.misc as misc

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None



# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
   assert(type(images) == list)
   assert(type(images[0]) == np.ndarray)
   assert(len(images[0].shape) == 3)
   assert(np.max(images[0]) > 10)
   assert(np.min(images[0]) >= 0.0)
   inps = []
   for img in images:
      img = img.astype(np.float32)
      inps.append(np.expand_dims(img, 0))
   bs = 100
   with tf.Session() as sess:
      preds = []
      n_batches = int(math.ceil(float(len(inps)) / float(bs)))
      for i in range(n_batches):
         sys.stdout.write(".")
         sys.stdout.flush()
         inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
         #print(np.asarray(inp).shape)
         inp = np.concatenate(inp, 0)
         pred = sess.run(softmax, {'ExpandDims:0': inp})
         preds.append(pred)
      preds = np.concatenate(preds, 0)
      scores = []
      for i in range(splits):
         part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
         kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
         kl = np.mean(np.sum(kl, 1))
         scores.append(np.exp(kl))
      return np.mean(scores), np.std(scores)



# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
                    o.shape_val = tf.TensorShape(new_shape)
            #o._shape = tf.TensorShape(new_shape)原来的
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  
   # get images
   import glob
   real_paths  = glob.glob('/home/fabbric/Research/ugan/datasets/underwater_imagenet/test/*.jpg')[:1800]
   ugan0_paths = glob.glob('/home/fabbric/Research/ugan/tests/test_images/ugan_0.0/*.png')[:1800]
   ugan1_paths = glob.glob('/home/fabbric/Research/ugan/tests/test_images/ugan_1.0/*.png')[:1800]
   cycle_paths = glob.glob('/home/fabbric/Research/CycleGAN-tensorflow/testA/*.png')[:1800]

   print(len(real_paths))
   print(len(ugan0_paths))
   print(len(ugan1_paths))
   print(len(cycle_paths))
   
   cycle_images = []
   ugan0_images = []
   ugan1_images = []
   real_images  = []

   for i in real_paths:
      i = misc.imread(i)
      i = misc.imresize(i, (256,256))
      real_images.append(i)
   
   for i in cycle_paths:
      cycle_images.append(misc.imread(i))
   
   for i in ugan0_paths:
      ugan0_images.append(misc.imread(i))
   
   for i in ugan1_paths:
      ugan1_images.append(misc.imread(i))

   _init_inception()
   
   real_mean_scores, real_std_scores = get_inception_score(real_images)
   print('Got real mean scores')
   print(real_mean_scores)
   print(real_std_scores)

   
   ugan0_mean_scores, ugan0_std_scores = get_inception_score(ugan0_images)
   print('Got ugan0 mean scores')
   print(ugan0_mean_scores)
   print(ugan0_std_scores)
   print('\n')
   
   ugan1_mean_scores, ugan1_std_scores = get_inception_score(ugan1_images)
   print('Got ugan1 mean scores')
   print(ugan1_mean_scores)
   print(ugan1_std_scores)
   print('\n')
   
   cycle_mean_scores, cycle_std_scores = get_inception_score(cycle_images)
   print('Got cycle mean scores')
   print(cycle_mean_scores)
   print(cycle_std_scores)
   print('\n')
   
