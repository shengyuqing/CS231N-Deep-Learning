# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:20:04 2017

@author: Administrator
"""
import numpy as np

class ConvNetArch_1(object):
    """
    my own covulution layers architecture
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=[100,100], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
        """
        Initialize a new network.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C,H,W = input_dim
        self.params['Wcrp1'] = weight_scale*np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['bcrp1'] = np.zeros((1,num_filters))
        self.params['gammacrp1'] =  np.ones(num_filters)
        self.params['betacrp1'] = np.zeros(num_filters)
        
        self.params['Wcrp2'] = weight_scale*np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['bcrp2'] = np.zeros((1,num_filters))
        self.params['gammacrp2'] =  np.ones(num_filters)
        self.params['betacrp2'] = np.zeros(num_filters)
        
        self.params['Wcrp3'] = weight_scale*np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['bcrp3'] = np.zeros((1,num_filters))
        self.params['gammacrp3'] =  np.ones(num_filters)
        self.params['betacrp3'] = np.zeros(num_filters)
        
        self.params['Wcr'] = weight_scale*np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['bcr'] = np.zeros((1,num_filters))
        self.params['gammacr'] =  np.ones(num_filters)
        self.params['betacr'] = np.zeros(num_filters)

        self.params['War1'] = weight_scale*np.random.randn(H*W*num_filters/(4*4*4),hidden_dim[0])
        self.params['bar1'] = np.zeros((1,hidden_dim[0]))
        self.params['gammaar1'] =  np.ones(hidden_dim[0])
        self.params['betaar1'] =  np.zeros(hidden_dim[0])
        
        self.params['War2'] = weight_scale*np.random.randn(hidden_dim[0], hidden_dim[1])
        self.params['bar2'] = np.zeros((1,hidden_dim[1]))
        self.params['gammaar2'] =  np.ones(hidden_dim[1])
        self.params['betaar2'] =  np.zeros(hidden_dim[1])
        
        self.params['Wa'] = weight_scale*np.random.randn(hidden_dim[1], num_classes)
        self.params['ba'] = np.zeros(num_classes)
        
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
        
        
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the convolutional network.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        
        Wcrp1, bcrp1, gammacrp1, betacrp1 = self.params['Wcrp1'], self.params['bcrp1'], self.params['gammacrp1'], self.params['betacrp1']
        Wcrp2, bcrp2, gammacrp2, betacrp2 = self.params['Wcrp2'], self.params['bcrp2'], self.params['gammacrp2'], self.params['betacrp2']
        Wcrp3, bcrp3, gammacrp3, betacrp3 = self.params['Wcrp3'], self.params['bcrp3'], self.params['gammacrp3'], self.params['betacrp3']
        Wcrp3, bcrp3, gammacrp3, betacrp3 = self.params['Wcrp3'], self.params['bcrp3'], self.params['gammacrp3'], self.params['betacrp3']
        Wcr, bcr, gammacr, betacr = self.params['Wcr'], self.params['bcr'], self.params['gammacr'], self.params['betacr']
        War1, bar1, gammaar1, betaar1 = self.params['War1'], self.params['bar1'], self.params['gammaar1'], self.params['betaar1']
        War2, bar2, gammaar2, betaar2 = self.params['War2'], self.params['bar2'], self.params['gammaar2'], self.params['betaar2']
        Wa, ba = self.params['Wa'], self.params['ba']
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        scores = None
        N = X.shape[0]
        out_crp1,cache_crp1 = conv_bn_relu_pool_forward(X, Wcrp1, bcrp1, gammacrp1, betacrp1, conv_param, pool_param, bn_param)
        out_crp2,cache_crp2 = conv_relu_pool_forward(out_crp1, Wcrp2, bcrp2, conv_param, pool_param)
        out_crp3,cache_crp3 = conv_relu_pool_forward(out_crp2, Wcrp3, bcrp3, conv_param, pool_param)
        out_crp3,cache_crp3 = conv_relu_pool_forward(out_crp2, Wcrp3, bcrp3, conv_param, pool_param)
        out_cr,cache_cr = conv_relu_forward(out_crp3, Wcr, bcr, conv_param)
        
        
        out_ar1,cache_ar1 = affine_relu_forward(out_cr, Wcr, bcr, )
        scores,cache3 = affine_forward(out2, W3, b3)
        
        
        
        
        
        
        
        
        