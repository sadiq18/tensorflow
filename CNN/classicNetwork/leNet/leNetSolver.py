#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:35:16 2018

@author: sadik
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import sys
import time
from datetime import datetime

from network.solver import Solver

class LeNetSolver(Solver):
    """
        LeNet Solver
    """
    
    def __init__(self, dataset, net, common_params, solver_params):
        #process params
        self.moment = float(solver_params['moment'])
        self.learning_rate = float(solver_params['learning_rate'])
        self.batch_size = int(common_params['batch_size'])
        self.width = int(common_params['image_size'])
        self.height = int(common_params['image_size'])
        self.train_dir = str(solver_params['train_dir'])
        self.max_iteration = str(solver_params['max_iteration'])
        
        self.dataset = dataset
        self.net = net
        
        self.construct_graph()
        
        
    def _train(self):
        """Train model
            Create an optimizer and apply to all trainable variables.
            
            Args:
                total_loss: Total loss from net.loss
                global_step: Integer Variable counting the number of training steps processed
                
            Returns:
                train_op: op for training
        """
        opt = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
        grads = opt.compute_gradients(self.total_loss)
        
        apply_gradient_op = opt.apply_gradient(grads, global_step=self.global_step)
        
        return apply_gradient_op
    
    def construct_graph(self):
        #construct graph
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 1))
        self.labels = tf.placeholder(tf.float32, (self.batch_size, 10))
        
        self.predicts = self.net.inference(self.images)
        self.total_loss = self.net.loss(self.predicts, self.labels)
        
        tf.summary.scalar('loss', self.total_loss)
        self.train_op = self._train()
        
    def solve(self):
        saver1 = tf.train.Saver(self.net.pretrained_collection, write_version=1)
        saver2 = tf.train.Saver(self.net.trainable_collection, write_version=1)
        
        init = tf.global_variables_initializer()
        
        summary_op = tf.summary.merge_all()
        
        with tf.Session() as sess:
            sess.run(init)
            
            saver1.restore(sess, self.pretrain_path)
            
            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            
            for step in xrange(self.max_iteration):
                start_time = time.time()
                
                np_images, np_labels = self.dataset.batch()
                np_images = np.asarray(np_images, dtype=np.float32)
                np_images = np_images / 255 * 2 -1
                np_labels = np.asarray(np_labels, dtype=np.float32)
                
                _, loss_value = sess.run([self.train_op, self.total_loss],
                                                 feed_dict={self.images:np_images, self.labels:np_labels})
                
                duration = time.time() - start_time
                
                assert not np.isnan(loss_value), 'Model diverge with loss = NaN'
                
                if step % 10 == 0:
                    num_examples_per_step = self.dataset.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    
                    print (format_str %(datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                    
                    sys.stdout.flush()
                    
                    if step % 100 == 0:
                        summary_str = sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels})
                        summary_writer.add_summary(summary_str, step)
                        
                    if step % 5000 == 0:
                        saver2.save(sess, self.train_dir + '/model.ckpt', global_step=step)
    