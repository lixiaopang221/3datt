import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

class Anet():
    def __init__(self, net_type, name = 'anet'):
        self.name = name
        self.net_type = net_type
        with open(f'./net{self.net_type}.conf','r') as file:
            params = file.read()
            self.params = eval(params)


    def inference(self, inputs, label):
        'inputs:NDHWC'
        params      = self.params
        att         = Attention(params['attention'], 'att')
        conv_1      = Conv3d(params['conv_1'], 'conv_1')
        block_1     = Resblock(params['block_1'], 'block_1')
        conv_last   = Conv3d(params['conv_last'], 'conv_last')
        
        with tf.variable_scope(self.net_type):
            x = att(inputs) #---hgh
            # x = att.channelwise_conv3d(inputs) #---hgh
            # x = inputs
            if self.net_type == 'tc' or self.net_type == 'wt':
                x = tf.concat([x, label], axis=-1)
            x = conv_1(x)
            x = block_1(x)
            x = block_1(x)
            x = block_1(x)
            x = conv_last(x)
            x = tf.nn.softmax(x)
            self.out = x


class Conv3d(object):
    def __init__(self, params, name='conv'):
        self.name           = name
        self.filters        = params['filters']
        self.kernel_size    = params['kernel_size']
        self.padding        = params['padding']
        self.dilation_rate  = params['dilation_rate']
        self.regularizer    = l2_regularizer(1e-7)

    def __call__(self,inputs):
        return self.conv3d(inputs)

    def conv3d(self,inputs):
        'inputs:NDHWC'
        x = tf.layers.conv3d(inputs, self.filters, self.kernel_size,
                             padding = self.padding,
                             dilation_rate = self.dilation_rate,
                             kernel_regularizer = self.regularizer,
                             bias_regularizer = self.regularizer)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        return x


class Resblock(object):
    def __init__(self, params, name = 'res'):
        self.name           = name
        self.filters        = params['filters']
        self.kernel_size    = params['kernel_size']
        self.padding        = params['padding']
        self.dilation_rate  = params['dilation_rate']
        self.conv_num       = params['conv_num']
        self.regularizer    = l2_regularizer(1e-7)

    def __call__(self,inputs):
        return self.resblock(inputs)

    def resblock(self,inputs):
        'inputs:NDHWC'
        x = inputs
        for i in range(self.conv_num):
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.layers.conv3d(x, self.filters, self.kernel_size, 
                                 padding = self.padding,
                                 dilation_rate = self.dilation_rate,
                                 kernel_regularizer = self.regularizer,
                                 bias_regularizer = self.regularizer)
        x = inputs + x
        return x


class Attention():
    def __init__(self, params, name = 'attention'):
        self.name           = name
        self.filters        = params['filters']
        self.kernel_size    = params['kernel_size']
        self.padding        = params['padding']
        self.r              = params['r']
        self.regularizer    = l2_regularizer(1e-7)


    def __call__(self, inputs):
        return self.attention(inputs)

    def attention(self, inputs):
        'inputs: NDHWC'
        # x = self.depthwise_conv2d(inputs)
        x = self.channelwise_conv3d(inputs)
        att_x = tf.unstack(x, axis = -1)
        att_x = tf.concat(att_x, axis = 1) 
        # att_vec = self.get_mean_of_top10(att_x)
        att_vec = tf.reduce_mean(att_x, axis=[2,3])
        att_depth = att_vec.get_shape()[-1]

        att_vec = tf.layers.dense(att_vec, int(int(att_depth) / self.r),
                                  kernel_regularizer = self.regularizer,
                                  bias_regularizer = self.regularizer)
        att_vec = tf.nn.relu(att_vec)
        att_vec = tf.layers.dense(att_vec, att_depth,
                                  kernel_regularizer = self.regularizer,
                                  bias_regularizer = self.regularizer)
        att_vec = tf.sigmoid(att_vec)

        att_vec = tf.split(att_vec, inputs.get_shape()[-1], axis = -1)
        att_vec = tf.stack(att_vec, axis = -1)
        att_vec = tf.expand_dims(att_vec,2)
        att_vec = tf.expand_dims(att_vec,3)
        x = x * att_vec
        return x
        
    def channelwise_conv3d(self, inputs):
        'inputs: NDHWC'
        outs = []
        for i in range(inputs.get_shape()[-1]):
            x =  tf.layers.conv3d(inputs[...,i:i + 1], self.filters, self.kernel_size,
                                  activation = tf.nn.relu,
                                  padding = self.padding,
                                  kernel_regularizer = self.regularizer,
                                  bias_regularizer = self.regularizer)
            x =  tf.layers.conv3d(x, 1, self.kernel_size,
                                  activation = tf.nn.relu,
                                  padding = self.padding,
                                  kernel_regularizer = self.regularizer,
                                  bias_regularizer = self.regularizer)
            outs.append(x)
        x = tf.concat(outs, -1)
        return x

    def depthwise_conv2d(self, inputs):
        'inputs: NDHWC, 3d代替2d卷积'
        x = tf.split(inputs, inputs.get_shape()[-1], axis = -1)
        x = tf.concat(x, axis = 1) #NLHW1, L = D*C
        x = tf.layers.conv3d(x, self.filters, self.kernel_size,
                             padding = self.padding,
                             kernel_regularizer = self.regularizer,
                             bias_regularizer = self.regularizer)
        x = tf.nn.relu(x)
        x = tf.layers.conv3d(x, 1, self.kernel_size,
                             padding = self.padding,
                             kernel_regularizer = self.regularizer,
                             bias_regularizer = self.regularizer)
        x = tf.nn.relu(x)
        x = tf.split(x, inputs.get_shape()[-1], axis = 1)
        x = tf.concat(x, axis = -1)
        return x
    
    def get_mean_of_top10(self, inputs):
        'inputs: [n,(d*c),h,w]'
        img_fla = tf.unstack(inputs, axis=-1)   #--- w*[n,(d*c),h]
        img_fla = tf.concat(img_fla, axis=-1)   #--- [n,(d*c),(h*w)]
        k = int(img_fla.get_shape()[-1])//10
        img_t10 = tf.nn.top_k(img_fla, k)[0]
        mean = tf.reduce_mean(img_t10, axis=-1) #--- w*[n,(d*c)]
        return mean
            
            
if __name__ == '__main__':
    x = tf.ones((2,10,10,8,4))
    net = Anet()
    y = net.inference(x)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        out = sess.run(y)
        print(out.shape)
        #print(out)