'''
Created on Fri Mar  6  2020

@author: aliv
'''
import numpy as np
import tensorflow as tf

import time

def stable_softmax(X):
  exps = tf.math.exp(X - tf.reduce_max(X, 1, keepdims=True))
  return exps / tf.reduce_sum(exps, 1, keepdims=True)
  
def cross_entropy(logits,labels):
  p = -tf.math.log(stable_softmax(logits))
  return tf.reduce_sum(tf.multiply(labels,p), 1)
  
# inputs shape: (?, rank_list)
# binary labels only!
def m_pairwise_logits(labels, logits, weights):
  if weights is None:
    y_w = labels
  else:
    y_w = labels * weights
  rank_list_size = labels.shape[1]
  paired_labels = []
  paired_logits = []
  masks = []
  offset = 1.0
  for i in range(rank_list_size-1):
    for j in range(i+1,rank_list_size):
      paired_labels.append(((labels[:,i:i+1]-labels[:,j:j+1]) + offset) / (2 * offset))
      paired_logits.append(logits[:,i:i+1] - logits[:,j:j+1])
      masks.append((labels[:,i:i+1]-labels[:,j:j+1]) * (y_w[:,i:i+1]-y_w[:,j:j+1]))
      
  return tf.concat(paired_labels, axis=1), tf.concat(paired_logits, axis=1), tf.concat(masks, axis=1)

# It is assumed that y_true is binary
def m_get_loss_function(loss_str):
  if loss_str == 'sigmoid':
    def sigmoid_lfn(y_true, y_pred, weights):
      if weights is None:
        y_w = y_true
      else:
        y_w = y_true * weights
        
      labels = (y_w) / tf.reduce_sum(y_w, axis=1, keepdims=True)
      y_pred -= tf.reduce_mean(y_pred, -1, keepdims=True)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_pred) * tf.reduce_sum(y_w,1)
#       loss = cross_entropy(logits=y_pred, labels=y_w)
      return tf.reduce_sum(loss) / tf.reduce_sum(y_true)
    
    return sigmoid_lfn 
  
  if loss_str == 'softmax':
    def softmax_lfn(y_true, y_pred, weights):
      if weights is None:
        y_w = y_true
      else:
        y_w = y_true * weights
        
      y_w += 1.e-12
      labels = (y_w) / tf.reduce_sum(y_w, axis=1, keepdims=True)
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_pred) * tf.reduce_sum(y_w,1)
#       loss = cross_entropy(logits=y_pred, labels=y_w)
      return tf.reduce_sum(loss) / tf.reduce_sum(y_true)
    
    return softmax_lfn
  
  if loss_str == 'pair_sigmoid':
    def pair_sigmoid_lfn(y_true, y_pred, weights):
      pair_labels, pair_logits, masks = m_pairwise_logits(labels=y_true, logits=y_pred, weights=weights)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=pair_labels, logits=pair_logits) * masks
      agg_loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(masks, axis=1)
      return tf.reduce_mean(agg_loss)
    return pair_sigmoid_lfn
  
  raise Exception('"{}" is not a supported loss function!'.format(loss_str))

def m_loss_from_str(compound_loss_str):
  loss_strs = str.split(compound_loss_str, '+')
  
  loss_fns = []
  for loss_str in loss_strs:
    elements = loss_str.split('*')
    if len(elements) > 2:
      raise TypeError('each component in loss string should be like "w*loss_fn" -> {}'.format(loss_str))
    if len(elements) == 2:
      w = float(elements[0])
      func = elements[1]
    else:
      w = 1.0
      func = elements[0]
    elements2 = func.split('(')
    if len(elements2) > 2:
      raise TypeError('each component in loss string should be like "loss_fn(param)" -> {}'.format(func))
    if len(elements2) == 2:
      param = float(elements2[1][:-1])
      func = elements2[0]
    else:
      param = 1.0
      func = elements2[0]
      
    loss_fns.append([w, m_get_loss_function(func), param])
      
  def loss_fn(y_true, y_pred, weights=None):
    loss = 0.0
    for w, func, param in loss_fns:
      loss += w * func(y_true=y_true, y_pred=param*y_pred, weights=weights)
    return loss
  
  return loss_fn


    