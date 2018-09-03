from __future__ import print_function

import os, math

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import sys

def _conv_block(net, bottom, name, num_output, use_relu=True, kernel_size=3, stride=1, pad=1, bn_prefix='', bn_postfix='/bn', 
    scale_prefix='', scale_postfix='/scale',depthwise=False,weight_filler='xavier'):
    if depthwise is False :
      conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, 
                    num_output=num_output,  pad=pad, bias_term=False, weight_filler=dict(type=weight_filler), bias_filler=dict(type='constant'))
    else :
      conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,group= num_output,engine=1,type='ConvolutionDepthwise',
                    num_output=num_output,  pad=pad, bias_term=False, weight_filler=dict(type=weight_filler), bias_filler=dict(type='constant'))        
    net[name] = conv

    bn_name = '{}{}{}'.format(bn_prefix, name, bn_postfix)
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
			'eps': 0.001,
			'moving_average_fraction': 0.999,
        }
    batch_norm = L.BatchNorm(conv, in_place=True, **bn_kwargs)
    net[bn_name] = batch_norm
    scale_kwargs = {
        'param': [
            dict(lr_mult=1, decay_mult=0),
            dict(lr_mult=2, decay_mult=0),],
        }
    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0),**scale_kwargs)
    sb_name = '{}{}{}'.format(scale_prefix, name, scale_postfix)
    net[sb_name] = scale

    if use_relu:
        out_layer = L.ReLU(scale, in_place=True)
        relu_name = '{}/relu'.format(name)
        net[relu_name] = out_layer
    else:
        out_layer = scale

    return out_layer

def _dense_block(net, from_layer, num_layers, growth_rate, name,bottleneck_width=4):

  x = from_layer
  growth_rate = int(growth_rate/2)

  for i in range(num_layers):
    base_name = '{}_{}'.format(name,i+1)
    inter_channel = int(growth_rate * bottleneck_width / 4) * 4
	
    cb1 = _conv_block(net, x, '{}/branch1a'.format(base_name), kernel_size=1, stride=1, 
                               num_output=inter_channel, pad=0)
    cb1 = _conv_block(net, cb1, '{}/branch1b'.format(base_name), kernel_size=3, stride=1, 
                               num_output=growth_rate, pad=1)
							   
    cb2 = _conv_block(net, x, '{}/branch2a'.format(base_name), kernel_size=1, stride=1, 
                               num_output=inter_channel, pad=0)
    cb2 = _conv_block(net, cb2, '{}/branch2b'.format(base_name), kernel_size=3, stride=1, 
                               num_output=growth_rate, pad=1)
    cb2 = _conv_block(net, cb2, '{}/branch2c'.format(base_name), kernel_size=3, stride=1, 
                               num_output=growth_rate, pad=1)

    x = L.Concat(x, cb1, cb2, axis=1)
    concate_name = '{}/concat'.format(base_name)
    net[concate_name] = x

  return x



def _transition_block(net, from_layer, num_filter, name, with_pooling=True):

  conv = _conv_block(net, from_layer, name, kernel_size=1, stride=1, num_output=num_filter, pad=0)

  if with_pooling:
    pool_name = '{}/pool'.format(name)
    #pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    pooling = L.Pooling(conv, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net[pool_name] = pooling
    from_layer = pooling
  else:
    from_layer = conv


  return from_layer



def _stem_block(net, from_layer, num_init_features):

  stem1 = _conv_block(net, net[from_layer], 'stem1', kernel_size=3, stride=2,
                           num_output=num_init_features, pad=1)
  stem2 = _conv_block(net, stem1, 'stem2a', kernel_size=1, stride=1,
                           num_output=int(num_init_features/2), pad=0)
  stem2 = _conv_block(net, stem2, 'stem2b', kernel_size=3, stride=2,
                           num_output=num_init_features, pad=1)
  stem1 = L.Pooling(stem1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
  net['stem/pool'] = stem1

  concate = L.Concat(stem1, stem2, axis=1)
  concate_name = 'stem/concat'
  net[concate_name] = concate

  stem3 = _conv_block(net, concate, 'stem3', kernel_size=1, stride=1, num_output=num_init_features, pad=0)

  return stem3

def PeleeNetBody(net, from_layer='data', growth_rate=32, block_config = [3,5,8,6], bottleneck_width=[1,2,4,4], num_init_features=32, init_kernel_size=3, use_stem_block=True):

    assert from_layer in net.tops.keys()

    # Initial convolution
    if use_stem_block:
      from_layer = _stem_block(net, from_layer, num_init_features)

    else:
      padding_size = init_kernel_size / 2
      out_layer = _conv_block(net, net[from_layer], 'conv1', kernel_size=init_kernel_size, stride=2,
                               num_output=num_init_features, pad=padding_size)
      net.pool1 = L.Pooling(out_layer, pool=P.Pooling.MAX, kernel_size=2, pad=0,stride=2)
      from_layer = net.pool1

    total_filter = num_init_features
    if type(bottleneck_width) is list:
        bottleneck_widths = bottleneck_width
    else:
        bottleneck_widths = [bottleneck_width] * 4

    for idx, num_layers in enumerate(block_config):
      from_layer = _dense_block(net, from_layer, num_layers, growth_rate, name='stage{}'.format(idx+1), bottleneck_width=bottleneck_widths[idx])
      total_filter += growth_rate * num_layers
      print(total_filter)
      if idx == len(block_config) - 1:
        with_pooling=False
      else:
        with_pooling=True

      from_layer = _transition_block(net, from_layer, total_filter,name='stage{}_tb'.format(idx+1), with_pooling=with_pooling)



    return net


def add_classify_header(net, classes=120):
  bottom = net.tops.keys()[-1]

  net.global_pool = L.Pooling(net[bottom], pool=P.Pooling.AVE, global_pooling=True) 

  net.classifier = L.InnerProduct(net.global_pool, num_output=classes, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))

  net.prob = L.Softmax(net.classifier)
  return net

def add_classify_loss(net, classes=120):
  bottom = net.tops.keys()[-1]

  net.global_pool = L.Pooling(net[bottom], pool=P.Pooling.AVE, global_pooling=True) 

  net.classifier = L.InnerProduct(net.global_pool, num_output=classes, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))

  net.train_loss = L.SoftmaxWithLoss(net.classifier, net.label,include={'phase':caffe.TRAIN})
  net.test_loss = L.SoftmaxWithLoss(net.classifier, net.label,include={'phase':caffe.TEST})
  net.accuracy = L.Accuracy(net.classifier, net.label)
  net.accuracy_top5 = L.Accuracy(net.classifier, net.label,top_k = 5)
  return net
def add_yolo_loss(net):
  bottom = net.tops.keys()[-1]

  net.conv_global = L.Convolution(net[bottom], kernel_size=1, stride=1,name='conv_global',
                    num_output=125,  pad=0, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
  
  net.RegionLoss = L.RegionLoss(net.conv_global,net.label,num_class=20,coords=4,num=5,softmax=1,jitter=0.2,rescore=0,object_scale=5.0,noobject_scale=1.0,class_scale=1.0,
					coord_scale=1.0,absolute=1,thresh=0.6,random=0,biases=[1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52], include={'phase':caffe.TRAIN})
  return net.conv_global
def add_yolo_detection(net,conv):

  net.YoloDetectionOutput = L.YoloDetectionOutput(conv,net.label,num_classes=20,coords=4,confidence_threshold=0.01,nms_threshold=.45
					,biases=[1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52],include={'phase':caffe.TEST})
  #net.RegionTestLoss = L.RegionLoss(conv,net.label,num_class=20,coords=4,num=5,softmax=1,jitter=0.2,rescore=0,object_scale=5.0,noobject_scale=1.0,class_scale=1.0,
  #					coord_scale=1.0,absolute=1,thresh=0.6,random=0,biases=[1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52], include={'phase':caffe.TEST})
def add_yolo_data_header(net):
  resize_kwargs = [
        dict(prob=0.1,resize_mode=1,height=416,width=416,interp_mode=[1,2,3,4,5]),
        dict(prob=0.1,resize_mode=1,height=448,width=448,interp_mode=[1,2,3,4,5]),
        dict(prob=0.1,resize_mode=1,height=480,width=480,interp_mode=[1,2,3,4,5]),
        dict(prob=0.1,resize_mode=1,height=512,width=512,interp_mode=[1,2,3,4,5]),
        dict(prob=0.1,resize_mode=1,height=544,width=544,interp_mode=[1,2,3,4,5]),
        dict(prob=0.1,resize_mode=1,height=576,width=576,interp_mode=[1,2,3,4,5]),
        dict(prob=0.1,resize_mode=1,height=608,width=608,interp_mode=[1,2,3,4,5]),
        dict(prob=0.1,resize_mode=1,height=384,width=384,interp_mode=[1,2,3,4,5]),
        dict(prob=0.1,resize_mode=1,height=352,width=352,interp_mode=[1,2,3,4,5]),
        dict(prob=0.1,resize_mode=1,height=320,width=320,interp_mode=[1,2,3,4,5]),
  ]
  net.data,net.label = L.AnnotatedData(name='data', data_param=dict(batch_size=6, backend=P.Data.LMDB,source='examples/VOC0712/VOC0712_trainval_lmdb'),
                             transform_param=dict(mean_value=[127.5,127.5,127.5],scale=1/127.5, mirror=False,resize_param=resize_kwargs,
                             emit_constraint=dict(emit_type=0),distort_param=dict(brightness_prob=0.5,brightness_delta=32.0,contrast_prob=0.5,contrast_lower=0.5,contrast_upper=1.5,hue_prob=0.5,
                             hue_delta=18.0,saturation_prob=0.5,saturation_lower=0.5,saturation_upper=1.5,random_order_prob=0.0),expand_param=dict(prob=0.5,max_expand_ratio=2.0)),
                             annotated_data_param=dict(yolo_data_type=1,yolo_data_jitter=0.3,label_map_file='data/VOC0712/labelmap_voc.prototxt'),ntop=2, include={'phase':caffe.TRAIN})    
  net.test_data,net.test_label = L.AnnotatedData(name='data', data_param=dict(batch_size=1, backend=P.Data.LMDB,source='examples/VOC0712/VOC0712_test_lmdb'),
                             transform_param=dict(mean_value=[127.5,127.5,127.5],scale=1/127.5, mirror=False,resize_param=dict(prob=1,resize_mode=1,height=416,width=416,interp_mode=[2])),
                             annotated_data_param=dict(batch_sampler=dict(),label_map_file='data/VOC0712/labelmap_voc.prototxt'), top=['data','label'],ntop=2, include={'phase':caffe.TEST}) 
def add_imagenet_data_header(net):
  #net.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
  source_lmdb = '/media/data/data/ilsvrc12_train_lmdb'
  #source_lmdb = 'examples/imagenet/ilsvrc12_train_lmdb'
  net.data, net.label = L.Data(name='data',batch_size=32, backend=P.Data.LMDB, source=source_lmdb,
                             transform_param=dict(crop_size=224,mean_value=[127.5,127.5,127.5],scale=1/127.5, mirror=True),
							 ntop=2 , include={'phase':caffe.TRAIN},image_data_param=dict(shuffle=True))
  net.test_data, net.test_label = L.Data(name='data',batch_size=10, backend=P.Data.LMDB, source='examples/imagenet/ilsvrc12_val_lmdb',
                             transform_param=dict(crop_size=224,mean_value=[127.5,127.5,127.5],scale=1/127.5, mirror=False),
  							 ntop=2 , top=['data','label'],include={'phase':caffe.TEST})	 

def _depthwise_block1(net, from_layer, name,num_layers=1,kernel_size=3,stride=1,num_output=32):

  x = from_layer
  for i in range(num_layers):
    padding_size = kernel_size / 2
    base_name = '{}_{}/dw'.format(name,i+1)
    cb1 = _conv_block(net, x, '{}'.format(base_name), kernel_size=kernel_size, stride=stride, num_output=num_output, pad=padding_size,depthwise=True,weight_filler='msra')
    base_name = '{}_{}/sep'.format(name,i+1)
    cb1 = _conv_block(net, cb1, '{}'.format(base_name), kernel_size=1, stride=stride, num_output=num_output, pad=0,depthwise=False,weight_filler='msra')

  return cb1
    
def _depthwise_block2(net, from_layer, name,num_layers=1,kernel_size=3,bottleneck_width=3,num_output=32,stride_step = 1):

  x = from_layer
  for i in range(num_layers):
    if i is not 0 :
      stride = 1     
    else :
      stride = stride_step
    num_out = num_output/bottleneck_width
    padding_size = kernel_size / 2
    base_name = '{}_{}/sep1'.format(name,i+1)
    cb1 = _conv_block(net, x, '{}'.format(base_name), kernel_size=1, stride=1, num_output=num_output, pad=0,depthwise=False,weight_filler='msra')
    base_name = '{}_{}/dw'.format(name,i+1)
    cb1 = _conv_block(net, cb1, '{}'.format(base_name), kernel_size=kernel_size, stride=stride, num_output=num_output, pad=padding_size,depthwise=True,weight_filler='msra')
    base_name = '{}_{}/sep2'.format(name,i+1)
    cb1 = _conv_block(net, cb1, '{}'.format(base_name), kernel_size=1, stride=1, num_output=num_out, pad=0,depthwise=False,weight_filler='msra')
    if i is not 0 :
      x = L.Eltwise(x, cb1)
      eltwise_name = '{}/eltwise'.format(base_name)
      net[eltwise_name] = x
    else :
      x = cb1
  return x
    
def add_mNasNet_Body(net, from_layer='data',init_kernel_size = 3,num_init_features=32) :
  assert from_layer in net.tops.keys()
  padding_size = init_kernel_size / 2
  out_layer = _conv_block(net, net[from_layer], 'conv1', kernel_size=init_kernel_size, stride=2,
                               num_output=num_init_features, pad=padding_size,weight_filler='msra')
  #net.pool1 = L.Pooling(out_layer, pool=P.Pooling.MAX, kernel_size=2, pad=0,stride=2)
  from_layer = out_layer
  idx = 2
  from_layer = _depthwise_block1(net, from_layer, name='conv{}'.format(idx),num_layers=1,num_output=16)
  idx+=1
  from_layer = _depthwise_block2(net, from_layer, name='conv{}'.format(idx),num_layers=3,num_output=72,stride_step=2)
  idx+=1
  from_layer = _depthwise_block2(net, from_layer, name='conv{}'.format(idx),num_layers=3,num_output=120,kernel_size=5,stride_step=2)
  idx+=1
  from_layer = _depthwise_block2(net, from_layer, name='conv{}'.format(idx),num_layers=3,num_output=480,kernel_size=5,stride_step=2,bottleneck_width=6)
  idx+=1
  from_layer = _depthwise_block2(net, from_layer, name='conv{}'.format(idx),num_layers=2,num_output=576,bottleneck_width=6)
  idx+=1
  from_layer = _depthwise_block2(net, from_layer, name='conv{}'.format(idx),num_layers=4,num_output=1152,kernel_size=5,stride_step=2,bottleneck_width=6)
  idx+=1
  from_layer = _depthwise_block2(net, from_layer, name='conv{}'.format(idx),num_layers=1,num_output=1920,bottleneck_width=6)
  idx+=1
if __name__ == '__main__':
  net = caffe.NetSpec()
  if(len(sys.argv)<2):
    print('Usage : python testnet.py [imagenet,yolo]')
    sys.exit()
  print(sys.argv[1])
  if(sys.argv[1] in 'imagenet'):
    add_imagenet_data_header(net)
  elif(sys.argv[1] in 'yolo'):
    add_yolo_data_header(net)
  else:
    add_imagenet_data_header(net)
  add_mNasNet_Body(net, from_layer='data')  
  #PeleeNetBody(net, from_layer='data')
  #add_classify_header(net,classes=1000)
  #add_classify_loss(net,classes=1000)
  if(sys.argv[1] in 'imagenet'):
    add_classify_loss(net,classes=1000)  
  elif(sys.argv[1] in 'yolo'):
    conv = add_yolo_loss(net)
    add_yolo_detection(net,conv)
  else:
    add_classify_loss(net,classes=1000)
  
  #print(net.to_proto())
# then write back out:
  with open('train_val.prototxt', 'w') as f:
    f.write(text_format.MessageToString(net.to_proto()))


