# coding:utf-8
import matplotlib.pyplot as plt
from trainer import *
from data_utils import *

class SRTrainer(CNNTrainer):
    """
    patch size = 21
    cnn depth = 10
    """
    def __init__(self, model_save_file="", model_load_file="", model_tag=0):
        CNNTrainer.__init__(self,model_save_file=model_save_file, model_load_file=model_load_file, model_tag=model_tag)


class SRTrainerDecon(CNNTrainer):
    """
    去卷积
    """
    def __init__(self, model_save_file="", model_load_file="", model_tag=0):
        CNNTrainer.__init__(self,model_save_file=model_save_file, model_load_file=model_load_file, model_tag=model_tag)

    def op_conv2d_trans(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape,  strides = [1, 1, 1, 1], padding='SAME')

    def inference(self):
        self.norm_list = []
        self.pool_list = []
        self.drop_list = []
        self.conv_list = []


        self.net_input_holder = tf.placeholder(tf.float32, shape=(None, self.input_size, self.input_size, self.input_depth))
        self.net_output_holder = tf.placeholder(tf.float32, shape=(None, self.input_size, self.input_size, self.input_depth))

        drop = tf.reshape(self.net_input_holder, [-1, self.input_size, self.input_size, self.input_depth])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('conv_front') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.input_depth, self.layer_depth], "weights")
            bias = self.variable_bias([self.layer_depth], "biases")
            conv = self.op_conv2d(drop, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(conv_out)

        norm = self.op_normalize(conv_out, self.USE_NORM)
        pool = self.op_max_pool_2x2(norm, self.USE_POOL)
        drop = self.op_dropout(pool, self.keep_prob)

        self.norm_list.append(norm)
        self.pool_list.append(pool)
        self.drop_list.append(drop)

        for i in range(self.layer_num):
            with tf.variable_scope('conv%d' % i) as scope:
                kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
                bias = self.variable_bias([self.layer_depth], "biases")
                conv = self.op_conv2d(drop, kernel)
                pre_activation = tf.nn.bias_add(conv, bias)
                conv_out = tf.nn.relu(pre_activation, name=scope.name)

                self.conv_list.append(conv_out)

            norm = self.op_normalize(conv_out, self.USE_NORM)
            pool = self.op_max_pool_2x2(norm, self.USE_POOL)
            drop = self.op_dropout(pool, self.keep_prob)

            self.norm_list.append(norm)
            self.pool_list.append(pool)
            self.drop_list.append(drop)

        with tf.variable_scope('conv_back') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
            bias = self.variable_bias([self.layer_depth], "biases")
            conv = self.op_conv2d(drop, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(conv_out)


        with tf.variable_scope('deconv') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, 1, self.layer_depth], "weights")
            bias = self.variable_bias([1], "biases")
            conv = self.op_conv2d_trans(conv_out, kernel,output_shape=[self.batch_size,21,21,1])
            pre_activation = tf.nn.bias_add(conv, bias)

            self.conv_list.append(pre_activation)

        self.net_output_calc = pre_activation


class SRTrainerDecon2(CNNTrainer):
    """
    去卷积
    """
    def __init__(self, model_save_file="", model_load_file="", model_tag=0):
        CNNTrainer.__init__(self,model_save_file=model_save_file, model_load_file=model_load_file, model_tag=model_tag)

    def op_conv2d_trans(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape,  strides = [1, 1, 1, 1], padding='SAME')

    def inference(self):
        self.norm_list = []
        self.pool_list = []
        self.drop_list = []
        self.conv_list = []


        self.net_input_holder = tf.placeholder(tf.float32, shape=(None, self.input_size, self.input_size, self.input_depth))
        self.net_output_holder = tf.placeholder(tf.float32, shape=(None, self.input_size, self.input_size, self.input_depth))

        drop = tf.reshape(self.net_input_holder, [-1, self.input_size, self.input_size, self.input_depth])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('conv_front') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.input_depth, self.layer_depth], "weights")
            bias = self.variable_bias([self.layer_depth], "biases")
            conv = self.op_conv2d(drop, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(conv_out)

        norm = self.op_normalize(conv_out, self.USE_NORM)
        pool = self.op_max_pool_2x2(norm, self.USE_POOL)
        drop = self.op_dropout(pool, self.keep_prob)

        self.norm_list.append(norm)
        self.pool_list.append(pool)
        self.drop_list.append(drop)

        for i in range(int(self.layer_num / 2)):
            with tf.variable_scope('conv%d' % i) as scope:
                kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
                bias = self.variable_bias([self.layer_depth], "biases")
                conv = self.op_conv2d(drop, kernel)
                pre_activation = tf.nn.bias_add(conv, bias)
                conv_out = tf.nn.relu(pre_activation, name=scope.name)

                self.conv_list.append(conv_out)

            norm = self.op_normalize(conv_out, self.USE_NORM)
            pool = self.op_max_pool_2x2(norm, self.USE_POOL)
            drop = self.op_dropout(pool, self.keep_prob)

            self.norm_list.append(norm)
            self.pool_list.append(pool)
            self.drop_list.append(drop)

        with tf.variable_scope('conv_back') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
            bias = self.variable_bias([self.layer_depth], "biases")
            conv = self.op_conv2d(drop, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(conv_out)

        for i in range(int(self.layer_num / 2)):
            with tf.variable_scope('deconv%d' % i) as scope:
                kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
                bias = self.variable_bias([self.layer_depth], "biases")
                conv = self.op_conv2d_trans(conv_out, kernel,output_shape=[self.batch_size,21,21,self.layer_depth])
                pre_activation = tf.nn.bias_add(conv, bias)
                conv_out = tf.nn.relu(pre_activation, name=scope.name)

                self.conv_list.append(pre_activation)

        with tf.variable_scope('deconv_back') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, 1, self.layer_depth], "weights")
            bias = self.variable_bias([1], "biases")
            conv = self.op_conv2d_trans(conv_out, kernel, output_shape=[self.batch_size, 21, 21, 1])
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(pre_activation)
        self.net_output_calc = pre_activation


class SRTrainerDecon3(CNNTrainer):
    """
    去卷积
    """
    def __init__(self, model_save_file="", model_load_file="", model_tag=0):
        CNNTrainer.__init__(self,model_save_file=model_save_file, model_load_file=model_load_file, model_tag=model_tag)

    def op_conv2d_trans(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape,  strides = [1, 1, 1, 1], padding='SAME')

    def inference(self):
        self.norm_list = []
        self.pool_list = []
        self.drop_list = []
        self.conv_list = []


        self.net_input_holder = tf.placeholder(tf.float32, shape=(None, self.input_size, self.input_size, self.input_depth))
        self.net_output_holder = tf.placeholder(tf.float32, shape=(None, self.input_size, self.input_size, self.input_depth))

        drop = tf.reshape(self.net_input_holder, [-1, self.input_size, self.input_size, self.input_depth])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('conv_front') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.input_depth, self.layer_depth], "weights")
            bias = self.variable_bias([self.layer_depth], "biases")
            conv = self.op_conv2d(drop, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(conv_out)

        norm = self.op_normalize(conv_out, self.USE_NORM)
        pool = self.op_max_pool_2x2(norm, self.USE_POOL)
        drop = self.op_dropout(pool, self.keep_prob)

        self.norm_list.append(norm)
        self.pool_list.append(pool)
        self.drop_list.append(drop)

        for i in range(1):
            with tf.variable_scope('conv%d' % i) as scope:
                kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
                bias = self.variable_bias([self.layer_depth], "biases")
                conv = self.op_conv2d(drop, kernel)
                pre_activation = tf.nn.bias_add(conv, bias)
                conv_out = tf.nn.relu(pre_activation, name=scope.name)

                self.conv_list.append(conv_out)

            norm = self.op_normalize(conv_out, self.USE_NORM)
            pool = self.op_max_pool_2x2(norm, self.USE_POOL)
            drop = self.op_dropout(pool, self.keep_prob)

            self.norm_list.append(norm)
            self.pool_list.append(pool)
            self.drop_list.append(drop)

        with tf.variable_scope('conv_back') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
            bias = self.variable_bias([self.layer_depth], "biases")
            conv = self.op_conv2d(drop, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(conv_out)

        for i in range(int(self.layer_num)):
            with tf.variable_scope('deconv%d' % i) as scope:
                kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
                bias = self.variable_bias([self.layer_depth], "biases")
                conv = self.op_conv2d_trans(conv_out, kernel,output_shape=[self.batch_size,self.input_size,self.input_size,self.layer_depth])
                pre_activation = tf.nn.bias_add(conv, bias)
                conv_out = tf.nn.relu(pre_activation, name=scope.name)

                self.conv_list.append(pre_activation)

        with tf.variable_scope('deconv_back') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, 1, self.layer_depth], "weights")
            bias = self.variable_bias([1], "biases")
            conv = self.op_conv2d_trans(conv_out, kernel, output_shape=[self.batch_size, self.input_size, self.input_size, 1])
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(pre_activation)
        self.net_output_calc = pre_activation

class SRTrainerDeconDeep(CNNTrainer):
    """
    去卷积
    """
    def __init__(self, model_save_file="", model_load_file="", model_tag=0):
        CNNTrainer.__init__(self,model_save_file=model_save_file, model_load_file=model_load_file, model_tag=model_tag)

    def op_conv2d_trans(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape,  strides = [1, 1, 1, 1], padding='SAME')

    def inference(self):
        self.norm_list = []
        self.pool_list = []
        self.drop_list = []
        self.conv_list = []


        self.net_input_holder = tf.placeholder(tf.float32, shape=(None, self.input_size, self.input_size, self.input_depth))
        self.net_output_holder = tf.placeholder(tf.float32, shape=(None, self.input_size, self.input_size, self.input_depth))

        drop = tf.reshape(self.net_input_holder, [-1, self.input_size, self.input_size, self.input_depth])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('conv_front') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.input_depth, self.layer_depth], "weights")
            bias = self.variable_bias([self.layer_depth], "biases")
            conv = self.op_conv2d(drop, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(conv_out)

        norm = self.op_normalize(conv_out, self.USE_NORM)
        pool = self.op_max_pool_2x2(norm, self.USE_POOL)
        drop = self.op_dropout(pool, self.keep_prob)

        self.norm_list.append(norm)
        self.pool_list.append(pool)
        self.drop_list.append(drop)

        for i in range(1):
            with tf.variable_scope('conv%d' % i) as scope:
                kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
                bias = self.variable_bias([self.layer_depth], "biases")
                conv = self.op_conv2d(drop, kernel)
                pre_activation = tf.nn.bias_add(conv, bias)
                conv_out = tf.nn.relu(pre_activation, name=scope.name)

                self.conv_list.append(conv_out)

            norm = self.op_normalize(conv_out, self.USE_NORM)
            pool = self.op_max_pool_2x2(norm, self.USE_POOL)
            drop = self.op_dropout(pool, self.keep_prob)

            self.norm_list.append(norm)
            self.pool_list.append(pool)
            self.drop_list.append(drop)

        with tf.variable_scope('conv_back') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
            bias = self.variable_bias([self.layer_depth], "biases")
            conv = self.op_conv2d(drop, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(conv_out)

        for i in range(int(self.layer_num)):
            with tf.variable_scope('deconv%d' % i) as scope:
                kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.layer_depth], "weights")
                bias = self.variable_bias([self.layer_depth], "biases")
                conv = self.op_conv2d_trans(conv_out, kernel,output_shape=[self.batch_size,self.input_size,self.input_size,self.layer_depth])
                pre_activation = tf.nn.bias_add(conv, bias)
                conv_out = tf.nn.relu(pre_activation, name=scope.name)

                self.conv_list.append(pre_activation)

        with tf.variable_scope('deconv_back') as scope:
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, 1, self.layer_depth], "weights")
            bias = self.variable_bias([1], "biases")
            conv = self.op_conv2d_trans(conv_out, kernel, output_shape=[self.batch_size, self.input_size, self.input_size, 1])
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(pre_activation)
        self.net_output_calc = pre_activation

if __name__ == "__main__":
    data_file_name = "E:/mySuperResolution/dataset/291/291_cnn_Y_channel_21_with_down.pic"
    training_data = load_training_data(data_file_name)
    print ("训练数据共有%d" % len(training_data[0]))
    input = training_data[0]
    output = training_data[1]

    with tf.Graph().as_default():
        tr = SRTrainerDecon3(model_save_file="E:/mySuperResolution/dataset/Y_291_decon_v3_with_drop_seg2",
                       model_load_file="E:/mySuperResolution/dataset/Y_291_decon_v3_with_drop-0",
                       model_tag=0)
        tr.init_param(layer_depth=32,layer_num=8, input_size=21,keep_prob_value = 0.85)
        tr.set_data(input,output)
        tr.setup_frame()
        tr.training()
        # tr.restoring()
