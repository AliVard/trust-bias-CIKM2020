'''
Created on Fri Mar  6  2020

@author: aliv


sbatch -c8 -n1 jobs/trustpbm.job
'''
from __future__ import print_function
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

import numpy as np
from absl import app
from absl import flags
import json

FLAGS = flags.FLAGS

import sys
import time
import os

SYS_PATH_APPEND_DEPTH = 2
SYS_PATH_APPEND = os.path.abspath(__file__)
for _ in range(SYS_PATH_APPEND_DEPTH):
  SYS_PATH_APPEND = os.path.dirname(SYS_PATH_APPEND)
sys.path.append(SYS_PATH_APPEND)
 
from trust_bias_EM import data_utils
from trust_bias_EM.RelevanceProbs import RelNet
from trust_bias_EM.metrics import eval_output_unbiased, eval_output, eval_output_unbiased_denoised

import logging

LAST_MODIFIED_TIME = 'Thu Mar 19 17:29:14 2020'

NON_DEFINED_FLAGS = set()
for k, _ in FLAGS.__flags.items():
  NON_DEFINED_FLAGS.add(k)


flags.DEFINE_string(  'gold_zetap', None, 'ground truth P(C=1 | R=1) per position.')
flags.DEFINE_string(  'gold_zetan', None, 'ground truth P(C=1 | R=0) per position.')
flags.DEFINE_string(  'correction', 'Affine', 'Correction method to use: "Nothing", "PBM", "Bayes", "Affine". (In case of "PBM", gold_zetap flag acts as gold_theta.)')
flags.DEFINE_boolean( 'learn_bias_params', True, 'Use EM to learn trust-bias parameters.')
flags.DEFINE_integer( 'steps_warming_EM', 4, 'steps for warming trust-bias parameters by EM. During these steps, EM is applied per mini-batch.')
flags.DEFINE_integer( 'steps_per_update_EM', 20, 'steps for updating trust-bias parameters by EM.')
flags.DEFINE_string(  'logits_to_prob_fn', 'combined', 'function for converting logits to probabilities: "min_max", "sigmoid" or "combined"')

flags.DEFINE_float(   'propensity_clip', 1000,
                      'propensity clip.')
    

flags.DEFINE_list(    'hidden_layer_size', [512, 256, 128], 
                      'list of hidden layer sizes')
flags.DEFINE_integer( 'batch_size', 32, 
                      'batch size')
flags.DEFINE_list(    'drop_out_probs', [0.0, 0.1, 0.1], 
                      'layer specific drop out probabilities. It has to have similar length with "hidden_layer_size" flag.')
flags.DEFINE_string(  'learning_rate', '4e-3', 
                      'learning rate')
flags.DEFINE_float(   'max_gradient_norm', 50.0, 
                      'if > 0, this value is used to clip the gradients.')
flags.DEFINE_float(   'l2_loss', 0.0, 
                      'used for regularization with l2 norm. This is the coefficient!')
flags.DEFINE_string(  'loss_fn', 'softmax',
                      'loss function for optimization: "softmax", "lambdaloss"')
flags.DEFINE_string(  'optimizer', 'adagrad', 
                      'which optimizer to use? Either "grad" for gradient descent or "adagrad" for adaptive gradient.')
flags.DEFINE_boolean( 'batch_normalize', False, 
                      'apply batch normalization at each layer')
flags.DEFINE_boolean( 'fresh', True, 
                      'set for ignoring the chekpoint model in "ckpt_dir" directory')

flags.DEFINE_string(  'ckpt_dir', '', 
                      'directory of check points train files (if any)')

    
# file addresses:
# features
flags.DEFINE_string(  'dataset_name', 'Webscope_C14_Set1', 
                      'name of dataset')
flags.DEFINE_string(  'datasets_info_path', 'trust_bias/preprocess/datasets_info.json', 
                      'path to the datasets info file.')
flags.DEFINE_integer( 'data_fold', 0, 
                      'data fold number')
# clicks
flags.DEFINE_string(  'clicks_info_path', 'trust_bias/preprocess/clicks_pickle_paths.json',
                      'path to the json file containing paths to click pickles.')
flags.DEFINE_string(  'clicks_count', '2**19',
                      'number of clicks used')
flags.DEFINE_string(  'click_policy_name', 'trust_1_top10',
                      'click policy used for simulating clicks.')
  
flags.DEFINE_string(  'file_log_path', 'file.log', 
                      'path for logging.')

# other params:
flags.DEFINE_integer( 'max_train_iteration', 0, 
                      'Limit on the iterations of training (0: no limit).')
flags.DEFINE_integer( 'steps_per_checkpoint', 20, 
                      'How many training steps to do per checkpoint.')
flags.DEFINE_string(  'steps_no_checkpoint', '20', 
                      'How many training steps to do before any checkpoint.')
flags.DEFINE_boolean( 'predict', False,  
                      'set for predicting test data using learned model in "ckpt_dir" directory')

flags.DEFINE_boolean( 'train_and_predict', True,  
                      'set for training on train data and then predicting test data using learned model in "ckpt_dir" directory')
flags.DEFINE_string(  'train_and_predict_output', 'train_and_predict_output.txt', 
                      'comma separated output file for hyper parameter tuning purposes.')
# flags.DEFINE_string(  'perplexity_output', 'perplexity_output.txt', 
#                       'comma separated output file for hyper parameter tuning purposes.')
# flags.DEFINE_string(  'perplexity_prob_fn', 'softmax', 
#                       'function used for converting logits to probs for perplexity. either "softmax", "min_max" or "sigmoid".')

flags.DEFINE_string(  'slurm_job_id', '0', 
                      'job id (and task id in case of array) from slurm')
flags.DEFINE_boolean( 'test_per_eval', True, 'debugging flag. do not set!')
# use sgd instead of adam
flags.DEFINE_string(  'eval_method', 'loss', 'either "metric" or "loss"')
flags.DEFINE_integer( 'nonimproving_steps', 50, 'stop training after this many checkpoints have been seen without validation improvement.')


DEFINED_FLAGS = set()
for k, _ in FLAGS.__flags.items():
  if not k in NON_DEFINED_FLAGS:
    DEFINED_FLAGS.add(k)

click_global_step = 0
rel_global_step = 0


def create_model(data_set, checkpoint_path, forward_only, logger=None):
  model = RelNet(    checkpoint_path=checkpoint_path, 
                     layers_size=FLAGS.hidden_layer_size,
                     embed_size=data_set.feature_matrix.shape[1], batch_size=FLAGS.batch_size,
                     drop_out_probs=FLAGS.drop_out_probs,
                     learning_rate=eval(FLAGS.learning_rate), 
                     loss_function_str=FLAGS.loss_fn,
                     max_gradient_norm=FLAGS.max_gradient_norm, l2_loss=FLAGS.l2_loss,
                     optimizer=FLAGS.optimizer, batch_normalize=FLAGS.batch_normalize,
                     gold_zetap=FLAGS.gold_zetap, 
                     gold_zetan=FLAGS.gold_zetan, 
                     correction=FLAGS.correction,
                     learn_bias_params=FLAGS.learn_bias_params,
                     propensity_clip=FLAGS.propensity_clip,
                     max_ranklist_size=data_set.max_ranklist_size,
                     logits_to_prob_fn=FLAGS.logits_to_prob_fn, 
                     logger=logger)
  return model

def train_model(  model_name, checkpoint_path, 
                  train_set, valid_set, test_set,
                  logger=None):

#   config = tf.ConfigProto()
#   config.gpu_options.allow_growth = True
  non_improving_checkpoint_steps = FLAGS.nonimproving_steps     
  
  model = create_model(data_set=train_set, checkpoint_path=checkpoint_path, forward_only=False, logger=logger)
  
  # Create model.
  logger.info("Creating %s model ..." % model_name)

  steps_no_checkpoint = eval(FLAGS.steps_no_checkpoint)
  # This is the training loop.
  step_time, loss = 0.0, 0.0
  current_step = model.global_step
  previous_losses = []
  previous_step_losses = []
  previous_valid_losses = []
  min_valid_loss = 0
  min_valid_loss_step = -1
  min_loss_set = False
#     best_loss = None
  while True:
    # Get a batch and make a step.
    start_time = time.time()
    
    step_loss = model.train_on_batch(train_set)
    

    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
    loss += step_loss / FLAGS.steps_per_checkpoint
    current_step += 1

    
    previous_step_losses.append(step_loss)
    if FLAGS.learn_bias_params:
      if current_step < FLAGS.steps_warming_EM or current_step % FLAGS.steps_per_update_EM == 0:
        logger.info("%s model: step %d loss %.9f" % 
                    (model_name, model.global_step, step_loss))
        model.update_trust_params(train_set)
        logger.info("params: step %d, %s" % (current_step, model._trust_params))
      
    # Once in a while, we save checkpoint, print statistics, and run evals.
    if current_step > steps_no_checkpoint and current_step % FLAGS.steps_per_checkpoint == 0:
      
      # Print statistics for the previous epoch.
      #loss = math.exp(loss) if loss < 300 else float('inf')
      logger.info("%s model: step %d step-time %.2f loss %.9f" % 
                  (model_name, model.global_step, step_time, loss))
      
      
      #train_writer.add_summary({'step-time':step_time, 'loss':loss}, current_step)

      # Decrease learning rate if no improvement was seen over last 3 times.
      #if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
      #  sess.run(model.learning_rate_decay_op)
      previous_losses.append(loss)
      
      if FLAGS.eval_method == 'metric':
        valid_loss = validate_model_by_metric(valid_set, model)
      elif FLAGS.eval_method == 'loss':
        valid_loss = validate_model_by_loss(valid_set, model)
      else:
        raise Exception('eval method not implemented!')
      
      
      logger.info("    eval: loss %.9f" % (valid_loss))
      if min_valid_loss_step == -1 or valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        min_valid_loss_step = current_step
        min_loss_set = True
      
      
      if test_set is not None:
        results = test_model(test_set, model, None, True)
        logger.info('Test nDCG@10: {}'.format(results))
#         logger.info('Test nDCG@10: {} ({} queries and {} docs.)'.format(results, test_set.test_doclist_ranges().shape[0]-1, test_set.test_labels().shape[0]))
      
      if loss == float('inf'):
        break

      step_time, loss = 0.0, 0.0
      sys.stdout.flush()

    
    if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
      logger.info('reached max train iteration: {} > {}'.format(current_step, FLAGS.max_train_iteration))
      break
    if min_valid_loss_step > 0 and current_step > (min_valid_loss_step + (non_improving_checkpoint_steps * FLAGS.steps_per_checkpoint)):
      logger.info('{} consecutive non improving checkpoints after {} '.format(non_improving_checkpoint_steps, min_valid_loss_step))
      break
        
    
  return model

def validate_model_by_loss(valid_set, model):
  # Validate model
  it = 0
  count_batch = 0.0
  valid_loss = 0

  while it < valid_set.samples_size - model.batch_size:
    v_loss = model.test_on_batch(valid_set, it)
    it += model.batch_size
    valid_loss += v_loss
    count_batch += 1.0
  if valid_set.samples_size - it > 0:
    v_loss = model.test_on_batch(valid_set, it)
    valid_loss += v_loss * model.batch_size / (valid_set.samples_size - it)
    count_batch += 1.0
    
  valid_loss /= count_batch
  
  return valid_loss

def validate_model_by_metric(valid_set, model):
  # Validate model
  it = 0
  count_batch = 0.0
  valid_loss = 0

  while it < valid_set.samples_size - model.batch_size:
    input_feed = model.get_seq_batch(it, valid_set)
    
    predicted = model.predict(input_feed)
    clicks = valid_set.y_train
    weights = model.get_InversePropensities(input_feed)
    
#     print('predicted:{}, clicks:{}, weights:{}'.format(predicted.shape, clicks.shape, weights.shape))
    result = eval_output_unbiased(y_true=clicks.reshape([-1]), 
                                  y_pred=predicted.reshape([-1]), 
                                  query_counts=FLAGS.topk, 
                                  weights=weights[0,:],
                                  k=10)
#     print('result:{}'.format(result[-1]))
#     print(predicted)
#     print(clicks)
#     print(result)
    it += model.batch_size
    valid_loss += result
    count_batch += 1.0
#   print('before: {}, batch={}, size={}, it={}'.format(valid_loss, count_batch, valid_set.samples_size, it))
  if valid_set.samples_size - it > 0:
    input_feed = model.get_seq_batch(it, valid_set)
    
    predicted = model.predict(input_feed)
    clicks = valid_set.y_train
    weights = model.get_InversePropensities(input_feed)
    
    result = eval_output_unbiased(y_true=clicks.reshape([-1]), 
                                  y_pred=predicted.reshape([-1]), 
                                  query_counts=FLAGS.topk, 
                                  weights=weights[0,:],
                                  k=10)
    valid_loss += result * model.batch_size / (valid_set.samples_size - it)
#     print('after: {}, result={} size={}, it={}'.format(valid_loss, result, valid_set.samples_size, it))
  
    count_batch += 1.0
    
  valid_loss /= count_batch
  
  return -valid_loss

def test_model(test_set, model, output_path, report_dcg=False):
  # This is the training loop.
  step_time = 0.0
  
  start_time = time.time()

  y_pred, y_true = model.predict(test_set)

  step_time += (time.time() - start_time)
  
  
  if output_path is not None:
    joint_ys = np.concatenate([np.expand_dims(y_pred, 1), np.expand_dims(y_true, 1)], axis=1)
    np.savetxt(output_path, joint_ys, fmt='%.5f, %.1f')
#   print('output saved in %s' % output_path)
  
  results = eval_output(y_true=y_true[:], y_pred=y_pred[:].numpy(), query_counts=np.diff(test_set.test_doclist_ranges()), report_dcg=report_dcg, k=10)
#   results = eval_output(y_true=y_true[:,0], y_pred=y_pred[:,0], query_counts=np.diff(test_set.doclist_ranges), report_dcg=report_dcg, k=10)
  
  return results

def train(click_data, logger):
  # Prepare data.
  train_set = click_data.train
  valid_set = click_data.valid
  test_set = None
  if FLAGS.test_per_eval:
    test_set = click_data.test
  checkpoint_path=os.path.join(FLAGS.ckpt_dir, "relNet.ckpt")
  if FLAGS.fresh:
    checkpoint_path = None
  rel_model = train_model(  model_name='relevance', 
                            checkpoint_path=checkpoint_path, 
                            train_set=train_set, valid_set=valid_set, test_set=test_set,
                            logger=logger) 

    
  rel_model.rel_global_step = rel_model.global_step
  
  return rel_model

def test(click_data, output_path, rel_model=None, logger=None):
  #   tf.debugging.set_log_device_placement(True)
  test_set = click_data.test

#   config = tf.ConfigProto()
#   config.gpu_options.allow_growth = True
 
  
  if rel_model is None:
    model = create_model(test_set, os.path.join(FLAGS.ckpt_dir, "relNet.ckpt"), True, logger)
    # Create model.
    logger.info("Creating model...")
  else:
    model = rel_model

  results = test_model(test_set, model, output_path, False)
  
  logger.info('Final result: Test nDCG@10: {} ({} queries and {} docs.)'.format(results, test_set.test_doclist_ranges().shape[0]-1, test_set.test_labels().shape[0]))
#   print(results)
  
  
  return results

      
def my_serialize(v):
  if v.value is not None:
    return v.serialize()
  else:
    return '--{}=None'.format(v.name)

def train_and_test(click_data, output_path, logger):
  rel_model = train(click_data=click_data,
                    logger=logger)
  results = test(click_data=click_data, 
                 output_path=output_path,
                 rel_model=rel_model,
                 logger=logger)
  
#   perp = perplexity(rel_model, logger=logger)
  
  separator_char = ''
  general_info = ''
  for k, v in FLAGS.__flags.items():
    if k in DEFINED_FLAGS:
      general_info += '{}"{}"'.format(separator_char, my_serialize(v)[2:])
      separator_char = ', '
    
  with open(FLAGS.train_and_predict_output, 'a') as fo:
    fo.write('{}, {}, "rel_global_step={}", {}, "nDCG={}"\n'.format(
                                                                FLAGS.slurm_job_id, 
                                                                general_info,
                                                                rel_model.rel_global_step, 
                                                                rel_model._trust_params,
                                                                results))
#   with open(FLAGS.perplexity_output, 'a') as fo:
#     fo.write('{}, {}, "rel_global_step={}", "perplexity={}"\n'.format(
#                                                                                     FLAGS.slurm_job_id, 
#                                                                                     general_info,
#                                                                                     rel_model.rel_global_step, 
#                                                                                     perp))
  
def main(args):
  
  logger = logging.getLogger('AliV')
  
  logger.info('Last modified: {}'.format(LAST_MODIFIED_TIME))
  
  f_handler = logging.FileHandler(FLAGS.file_log_path)
  f_handler.setLevel(logging.DEBUG)
  f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  f_handler.setFormatter(f_format)
  
  logger.addHandler(f_handler)

  general_info = '\ntensorflow {}\n\n'.format(tf.__version__)
  
  for k, v in FLAGS.__flags.items():
    if k in DEFINED_FLAGS:
      general_info += '    {}\n'.format(my_serialize(v)) #''    - {} : {}\n'.format(k,v.value)

  general_info += '\n\n'
  
  logger.info(general_info)
  
  with open(FLAGS.clicks_info_path, 'r') as f:
    clicks_info = json.load(f)
  
  click_data = data_utils.read_click_data(dataset_name = FLAGS.dataset_name, 
                                          datasets_info_path = FLAGS.datasets_info_path, 
                                          data_fold = FLAGS.data_fold, 
                                          clicks_path = clicks_info[FLAGS.click_policy_name][FLAGS.clicks_count])
  
  output_path = os.path.join(FLAGS.ckpt_dir,'rel_predictions_' + os.path.splitext(os.path.basename(clicks_info[FLAGS.click_policy_name][FLAGS.clicks_count]))[0] + '.txt')
  
  logger.info('"data_stats": [\n\t"train":\n\t\t ["queries":{}, "docs":{}], '.format(click_data.train.doclist_ranges.shape[0] - 1, click_data.train.label_vector.shape[0]) + 
  '\n\t"valid":\n\t\t ["queries":{}, "docs":{}], '.format(click_data.valid.doclist_ranges.shape[0] - 1, click_data.valid.label_vector.shape[0]) + 
  '\n\t"test":\n\t\t ["queries":{}, "docs":{}]\n] '.format(click_data.test.doclist_ranges.shape[0] - 1, click_data.test.label_vector.shape[0]))
  if FLAGS.predict:
    test(click_data=click_data,
         rel_model=None, 
         output_path=output_path,
         logger=logger)
  elif FLAGS.train_and_predict:
    train_and_test(click_data=click_data,
                   output_path=output_path,
                   logger=logger)
  else:
    train(click_data=click_data,
          logger=logger)
  
if __name__ == '__main__':
  app.run(main)
