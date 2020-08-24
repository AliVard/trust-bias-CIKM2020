'''
Created on 14 Apr 2020

@author: aliv
'''

import numpy as np
import os
import sys


def test(name, a, mask, logger):
  pass
  # if logger is not None:
    
  #   logger.info('{} -> {}'.format(name, str(np.sum(a*mask,0)/np.sum(mask,0)).replace('\n',' ').replace(' ',',').replace(',,',',')))


class trust_params:
  def __init__(self, zetap, zetan, correction, learnable, logger=None):
    self.zetap = zetap
    self.zetan = zetan
    self.learnable = learnable
    self.logger = logger
    if learnable:
      self.zetap -= 1.e-3
      self.zetan += 1.e-3
    
    if correction == 'PBM':
      self.correct = lambda click_rates : click_rates / self.zetap
    elif correction == 'Bayes':
      self.correct = lambda click_rates : click_rates / (self.zetap + self.zetan)
    elif correction == 'Affine':
      def correct(click_rates):
        mean_rates = np.mean(click_rates,0)
        zetan = np.minimum(mean_rates, self.zetan)
        denom = np.array(self.zetap - zetan)
        denom[denom==0.] = 1.
        return (click_rates - zetan) / denom
      self.correct = correct
#       self.correct = lambda click_rates : (click_rates - self.zetan) / (self.zetap - self.zetan)
    else:
      raise Exception('correction method {} not supported.'.format(correction))
    

  def E(self, rel_probs):
    r = rel_probs
    c = self.zetap * r + self.zetan * (1. - r)
    rc_dim = (2,2)
    rc_dim += r.shape
    rc = np.zeros(rc_dim, dtype=np.float64)
    
#     rc[1,1] > rc[1,0] -> zp / c > 1 > (1 - zp) / (1 - c)
    denom = np.array(1. - c)
    denom[denom == 0.] = 1.
    rc[1,0,:] = (r * (1. - self.zetap)) / denom
    rc[0,0,:] = 1. - rc[1,0,:]
    denom = np.array(c)
    denom[denom == 0.] = 1.
    rc[1,1,:] = r * self.zetap / denom
    rc[0,1,:] = 1. - rc[1,1,:]
    
    return rc
    
  def update(self, click_rates, rel_probs, mask):
    c = click_rates
    rc = self.E(rel_probs)
    
    test('click_rates', c, mask, self.logger)
    test('rel_probs', rel_probs, mask, self.logger)
    test('rc00', rc[0,0,:], mask, self.logger)
    test('rc01', rc[0,1,:], mask, self.logger)
    test('rc10', rc[1,0,:], mask, self.logger)
    test('rc11', rc[1,1,:], mask, self.logger)
    
#     rc[1,1] > c * rc[1,1] + (1-c) * rc[1,0] -> rc[1,1] > rc[1,0]
    nom = np.sum(c * rc[1,1,:] * mask, axis=0)
    denom = nom + np.sum(((1. - c) * rc[1,0,:] * mask), axis=0)
    denom[denom==0.] = 1.
    self.zetap = nom / denom
    
#     rc[0,1] < c * rc[0,1] + (1-c) * rc[0,0] -> rc[0,1] < rc[0,0]
    nom = np.sum(c * rc[0,1,:] * mask, axis=0)
    denom = nom + np.sum(((1. - c) * rc[0,0,:] * mask), axis=0)
    denom[denom==0.] = 1.
    self.zetan = nom / denom
    
#     new_labels = self.correct(click_rates, rel_probs, mask)
#     new_labels = mask * (click_rates * rc[1,1,:] + (1. - click_rates) * rc[1,0,:])
#     test('new_labels', new_labels, self.)
#     
#     return new_labels
  
    
  def __str__(self):
    s = '"zetap='
    separator = '['
    for p in self.zetap:
      s += separator + str(p)
      separator = ','
    s += ']", "zetan='
    separator = '['
    for n in self.zetan:
      s += separator + str(n)
      separator = ','
    return s + ']"'
    
class trust_params_old:
  def __init__(self, zetap, zetan, correction, learnable):
    self.zetap = zetap
    self.zetan = zetan
    self.learnable = learnable
    
    if learnable:
      self.zetap -= 1.e-3
      self.zetan += 1.e-3
    
    if correction == 'PBM':
      self.correct = lambda click_rates : click_rates / self.zetap
    elif correction == 'Bayes':
      self.correct = lambda click_rates : click_rates / (self.zetap + self.zetan)
    elif correction == 'Affine':
      self.correct = lambda click_rates : (click_rates - self.zetan) / (self.zetap - self.zetan)
    else:
      raise Exception('correction method {} not supported.'.format(correction))
    

  def E(self, rel_probs):
    r = rel_probs
    c = self.zetap * r + self.zetan * (1. - r)
    rc_dim = (2,2)
    rc_dim += r.shape
    rc = np.zeros(rc_dim, dtype=np.float64)
    denum = np.array(1. - c)
    mask = denum == 0.
    denum[mask] = 1.
    rc[1,0,:] = (r * (1. - self.zetap)) / denum
    rc[0,0,:] = 1. - rc[1,0,:]
    denum = np.array(c)
    mask = denum == 0.
    denum[mask] = 1.
    rc[1,1,:] = r * self.zetap / denum
    rc[0,1,:] = 1. - rc[1,1,:]
    
    return rc
  def reset_piles(self):
    self.zetap_num = np.zeros_like(self.zetap)
    self.zetap_denum = np.zeros_like(self.zetap)
    self.zetan_num = np.zeros_like(self.zetan)
    self.zetan_denum = np.zeros_like(self.zetan)
    
  def pile_for_update(self, click_rates, rel_probs, mask):
    if not self.learnable:
      return
    c = click_rates
    rc = self.E(rel_probs)
    
    num = c * rc[1,1,:] * mask
    denum = num + ((1. - c) * rc[1,0,:] * mask)
    self.zetap_num += np.sum(num, axis=0)
    self.zetap_denum += np.sum(denum, axis=0)
    
    num = c * rc[0,1,:] * mask
    denum = num + ((1. - c) * rc[0,0,:] * mask)
    self.zetan_num += np.sum(num, axis=0)
    self.zetan_denum += np.sum(denum, axis=0)
    
  def update(self):
    self.zetap_denum[self.zetap_denum==0.]=1.
    self.zetap = self.zetap_num / self.zetap_denum
    self.zetan_denum[self.zetan_denum==0.]=1.
    self.zetan = self.zetan_num / self.zetan_denum
    
  def __str__(self):
    s = '"zetap='
    separator = '['
    for p in self.zetap:
      s += separator + str(p)
      separator = ','
    s += ']", "zetan='
    separator = '['
    for n in self.zetan:
      s += separator + str(n)
      separator = ','
    return s + ']"'
    