'''
Created on Fri Mar  6  2020

@author: aliv
'''

from __future__ import print_function
# from future.utils import raise_

import numpy as np
import tensorflow as tf

import os
import sys
from _ast import Or

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trust_bias.Losses import m_loss_from_str
from trust_bias import tf_losses

import logging


def sigmoid_prob(logits):
  return 1.*tf.sigmoid(1.*(logits - tf.reduce_mean(logits, -1, keepdims=True)))

def min_max_prob(logits):
  e = tf.exp(logits)
  e = e - tf.reduce_min(e,-1,keepdims=True)
  return 1. * e / tf.reduce_max(e, -1, keepdims=True)

def get_gold_trust_params(gold_inverse_propensities, gold_noises, propensity_clip, max_ranklist_size):
  gold_inverse_propensities = np.array(list(map(float, eval(gold_inverse_propensities))), dtype=np.float64) if gold_inverse_propensities is not None else np.ones(1, dtype=np.float64)
  gold_inverse_propensities = np.minimum(gold_inverse_propensities, np.ones_like(gold_inverse_propensities)*propensity_clip)
  gold_noises = np.array(list(map(float, eval(gold_noises)))) if gold_noises is not None else np.zeros(1, dtype=np.float64)
  
  # extend inverse_propensities and noises to "max_ranklist_size". fill with the last element
  self_gold_inverse_propensities = np.zeros(max_ranklist_size) + gold_inverse_propensities[-1]
  min_len = np.minimum(len(gold_inverse_propensities), max_ranklist_size)
  self_gold_inverse_propensities[:min_len] = gold_inverse_propensities[:min_len]
  self_gold_noises = np.zeros(max_ranklist_size) + gold_noises[-1]
  min_len = np.minimum(len(gold_noises), max_ranklist_size)
  self_gold_noises[:min_len] = gold_noises[:min_len]
  
  return self_gold_inverse_propensities, self_gold_noises
    
class RelNet:
  def __init__(self, 
               checkpoint_path, 
               layers_size, embed_size, batch_size,
               forward_only, drop_out_probs,
               learning_rate, learning_rate_decay_factor, 
               loss_function_str, 
               max_gradient_norm, l2_loss,
               optimizer, batch_normalize,
               gold_inverse_propensities, 
               gold_noises,
               propensity_clip,
#                perplexity_prob_fn,
               max_ranklist_size,
               logger=None):

    self._drop_out_probs = np.zeros(len(layers_size), dtype=np.float64)
    for i in range(min(len(layers_size),len(drop_out_probs))):
      self._drop_out_probs[i] = drop_out_probs[i]

    self._loss_function_str = loss_function_str
    self._hidden_layer_sizes = list(map(int, layers_size))
    self._learning_rate_decay_factor = learning_rate_decay_factor
    self._embed_size = embed_size
    self._max_gradient_norm = max_gradient_norm
    self._gold_inverse_propensities, self._gold_noises = get_gold_trust_params(gold_inverse_propensities, 
                                                                               gold_noises, 
                                                                               propensity_clip, 
                                                                               max_ranklist_size)
    
    self.batch_size = batch_size
    self.max_ranklist_size = max_ranklist_size

#     self.network_model = self.network(batch_normalize)
    self.build_network()
    
    if optimizer == 'adagrad':
#       self._optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
      self._optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'sgd':
      self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
      self._optimizer = tf.keras.optimizers.get(optimizer)
        
    if self._loss_function_str == 'softmax':
      self._loss_class = tf_losses.SoftmaxLoss(self.max_ranklist_size)
    elif self._loss_function_str == 'lambdaloss':
      self._loss_class = tf_losses.LambdaLoss(self.max_ranklist_size)
      
    
#     self.network_model.compile(optimizer=opt, loss=self._loss_class.loss_fn())
    self.global_step = 0
    
    
    if checkpoint_path is not None and os.path.exists(os.path.join(os.path.dirname(checkpoint_path), 'checkpoint')):
      logger.info('loading weights from {}'.format(checkpoint_path))
      self.network_model.load_weights(checkpoint_path)
      self.global_step = 1000
      
      
      
      
  
#       self.tf_feature_matrix  = tf.placeholder(tf.float64, shape=(None, self._embed_size), name='trust/feature_matrix')
#       self.tf_click_rates  = tf.placeholder(tf.float64, shape=(None), name='trust/click_rates')
#       self.tf_padding_mask = tf.placeholder(tf.float64, shape=(None), name='trust/padding_mask')
#       self.ltf_dropout_rate = []
#       for dropout_rate in self._drop_out_probs:
#         self.ltf_dropout_rate.append(tf.placeholder_with_default(dropout_rate, shape=()))
      
#       params = tf.trainable_variables()
  
  def build_network(self):
    current_size = self._embed_size
    self.ltf_w = []
    self.ltf_b = []
    
    for layer in range(len(self._hidden_layer_sizes)):
      fan_in = current_size
      fan_out = self._hidden_layer_sizes[layer]
#       glorot_uniform_initializer as is default in tf.get_variable()
      r = np.sqrt(6.0/(fan_in+fan_out))
#       tf_w = tf.Variable(tf.random_normal([current_size, self._hidden_layer_sizes[layer]], stddev=0.1), name='rel/w_{}'.format(layer))
      self.ltf_w.append(tf.Variable(tf.random.uniform([fan_in, fan_out], minval=-r, maxval=r, dtype=tf.float64), name='trust/w_{}'.format(layer)))
      self.ltf_b.append(tf.Variable(tf.constant(0.1, shape=[fan_out], dtype=tf.float64), name='trust/b_{}'.format(layer)))

      current_size = self._hidden_layer_sizes[layer]
    
    
    # Output layer
  
    fan_in = self._hidden_layer_sizes[-1]
    fan_out = 1
    r = np.sqrt(6.0/(fan_in+fan_out))
    self.ltf_w.append(tf.Variable(tf.random.uniform([fan_in, fan_out], minval=-r, maxval=r, dtype=tf.float64), name='rel/w_last'))
    self.ltf_b.append(tf.Variable(tf.constant(0.1, shape=[fan_out], dtype=tf.float64), name='rel/b_{}'.format('last')))


  @tf.function
  def network(self,
              feature_matrix,
              training):
    
    tf_output = feature_matrix
    
    for layer in range(len(self._hidden_layer_sizes)):
      
      tf_w = self.ltf_w[layer]
      tf_b = self.ltf_b[layer]
      # x.w+b
      tf_output_tmp = tf.nn.bias_add(tf.matmul(tf_output, tf_w, name='trust/mul_{}'.format(layer)), tf_b, name='trust/affine_{}'.format(layer))
      # activation: elu
      tf_output = tf.nn.elu(tf_output_tmp, name='trust/elu_{}'.format(layer))
      
#       generalization: drop_out
      if self._drop_out_probs[layer] > 0.0 and training:
#         With probability rate elements are set to 0. 
#         The remaining elemenst are scaled up by 1.0 / (1 - rate), so that the expected value is preserved.
        tf_output = tf.nn.dropout(tf_output, rate=self._drop_out_probs[layer], name = 'rel/drop_out_{}'.format(layer))

    
    # Output layer
    tf_w = self.ltf_w[-1]
    tf_b = self.ltf_b[-1]
    
    tf_output = tf.nn.bias_add(tf.matmul(tf_output, tf_w), tf_b, name='rel/affine_last')
    
    return tf_output


  def train_on_batch(self, mDataset):
    indexes = mDataset.get_random_indexes(self.batch_size)
    feature_matrix, click_rates, padding_mask = mDataset.load_batch(indexes)
    
    labels = np.reshape(click_rates, [-1, self.max_ranklist_size])
    mask = np.reshape(padding_mask, [-1, self.max_ranklist_size])
    inverse_propensities = self._gold_inverse_propensities
    noises = self._gold_noises
    
    self._loss_class.set_mask(mask)
      
#     loss = self.network_model.train_on_batch(x=feature_matrix, 
#                                              y=np.reshape(mask * (labels - noises) * inverse_propensities, [-1]))

#     y_pred = self.network_model(feature_matrix, training=True)
#     loss = lambda: self._loss_class.loss_fn()(y_true=np.reshape(mask * (labels - noises) * inverse_propensities, [-1]),
#                                               y_pred=y_pred)
#     print(y_pred)
#     print(loss())
    
    with tf.GradientTape() as tape:
      loss = self._loss_class.loss_fn()(y_true=mask * (labels - noises),
                                    y_pred=self.network(feature_matrix, training=True),
#                                     y_pred=self.network_model(feature_matrix, training=True),
                                    weights=inverse_propensities)
    
#     self._optimizer.minimize(loss, lambda: self.network_model.trainable_weights)

      params = self.ltf_w + self.ltf_b
#       print(params)
      gradients = tape.gradient(loss, params)
      if self._max_gradient_norm > 0:
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
        self._optimizer.apply_gradients(zip(clipped_gradients, params))
      else:
        self._optimizer.apply_gradients(zip(self.gradients, params))
          
          
    self.global_step += 1
    return loss
    
  def test_on_batch(self, mDataset, seq_index):
    if seq_index >= mDataset.samples_size:
      raise Exception('sequential batch start index exceeds total number of samples!')
  
    start = seq_index
    seq_index += self.batch_size
    end = seq_index if seq_index <= mDataset.samples_size else mDataset.samples_size
    
    
    indexes = np.array(list(range(start,end)))
    feature_matrix, click_rates, padding_mask = mDataset.load_batch(indexes)
    
    labels = np.reshape(click_rates, [-1, self.max_ranklist_size])
    mask = np.reshape(padding_mask, [-1, self.max_ranklist_size])
    inverse_propensities = self._gold_inverse_propensities
    noises = self._gold_noises
    
#     y_pred = self.network_model(feature_matrix, training=False)
    
    self._loss_class.set_mask(mask)
    loss = self._loss_class.loss_fn()(y_true=mask * (labels - noises),
                                          y_pred=self.network(feature_matrix, training=False),
#                                           y_pred=self.network_model(feature_matrix, training=False),
                                          weights=inverse_propensities)
    return loss

  def predict(self, mDataset):
    it = 0
    
    feature_matrix, labels = mDataset.load_test_epoch()
    l_out = []

    while it < feature_matrix.shape[0]:
      end = it + (self.batch_size  * self.max_ranklist_size)
      if end > feature_matrix.shape[0]:
        end = feature_matrix.shape[0]
      input = feature_matrix[it:end,:]
      out = self.network(input, training=False)
      
      out = tf.reshape(out,[-1])
      l_out.append(out)
      it = end
      
    bin_labels = np.zeros_like(labels, dtype=np.float64)
    bin_labels[labels>2] = 1.
    return np.concatenate(l_out, 0), bin_labels
  
  def update_learning_rate(self, min_learning_rate, const_learning_rate_steps):
    pass
  
  def decay_learning_rate(self, decay):
    pass

  def learning_rate(self):
    return 10.
#     return self._optimizer.learning_rate.numpy()
  
  def save_model(self, checkpoint_path):
    pass









  