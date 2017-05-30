# coding:utf-8
import tensorflow as tf
import super_restore as sr
from data_utils import *

class CNNTrainer(object):
    def __init__(self, model_save_file="", model_load_file="", model_tag=0):

        self.model_save_file = model_save_file
        self.model_load_file = model_load_file
        self.model_tag = model_tag

    def set_data(self, input, output):
        """
        这里的输入数据必须是（patch_number, patch_size, patch_size）的格式,
        下面的input_depth指的是通道数，在初始化参数的死后就需要确定。
        :param input:
        :param output:
        :return:
        """
        self.patch_in = np.reshape(input, (-1, self.input_size, self.input_size, self.input_depth))
        self.patch_out = np.reshape(output, (-1, self.input_size, self.input_size, self.input_depth))
        self.data_size = self.patch_in.shape[0]

    def set_test_data(self, input ,output):
        self.test_in = np.reshape(input, (-1, self.input_size, self.input_size, self.input_depth))
        self.test_out = np.reshape(output, (-1, self.input_size, self.input_size, self.input_depth))
        self.test_data_size = self.test_in.shape[0]

    def init_param(self, layer_depth=32, layer_num=8, input_size=21, input_depth=1 ,
                  kernel_size=3, keep_prob_value = 1.0, USE_POOL = False, USE_NORM = False,
                   reload = False,batch_size = 64,itr_num=50, test_dir="", test_result_pefix="",
                   learning_rate=1e-4):

        self.layer_depth = layer_depth
        self.layer_num = layer_num
        self.input_size = input_size
        self.input_depth = input_depth
        self.kernel_size = kernel_size
        self.USE_POOL = USE_POOL
        self.USE_NORM = USE_NORM
        self.learning_rate = learning_rate
        self.keep_prob_value = keep_prob_value
        self.reload = reload
        self.itr_num = itr_num
        self.batch_size = batch_size
        self.test_dir = test_dir
        self.test_result_pefix=test_result_pefix

    def setup_frame(self):
        self.sess = tf.InteractiveSession()
        self.inference()
        self.evaluation()
        self.loss()

    def restoring(self):

        result = [p for p in self.test_in]
        for i in range(0, self.test_data_size, self.batch_size):
            X = self.test_in[i:i+self.batch_size]
            Y = self.test_out[i:i+self.batch_size]
            feed_dict = {self.net_input_holder: X, self.net_output_holder: Y, self.keep_prob: 1.0}
            result[i:i+self.batch_size] = self.sess.run([self.net_output_calc], feed_dict=feed_dict)[0][:]
        return result

    def training(self):

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2_loss)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if self.reload is True:
            self.saver.restore(self.sess, self.model_load_file)

        for i in range(self.itr_num):
            loss_list = []
            for step in range(self.data_size // self.batch_size):
                offset = step * self.batch_size
                X = self.patch_in[offset:offset+self.batch_size]
                Y = self.patch_out[offset:offset+self.batch_size]

                feed_dict = {self.net_input_holder: X, self.net_output_holder: Y, self.keep_prob: self.keep_prob_value}

                _, loss_value = self.sess.run([train_step, self.l2_loss], feed_dict=feed_dict)
                del X, Y

                loss_list.append(loss_value)
                if step % 100 == 0 and step > 0:
                    print("[epoch %2.4f] loss\t%.4f" % (i + (float(step) * self.batch_size / len(self.patch_in)), loss_value))

            self.saver.save(self.sess, self.model_save_file, global_step=self.model_tag)

            logfile = open(self.model_save_file + ".csv", 'a')
            writer = csv.writer(logfile)
            error_list = sr.restore_dir(self.test_dir, self, self.test_result_pefix)
            error_list.append(np.mean(loss_list))
            writer.writerow(error_list)
            logfile.close()

    '''
    操作简化函数
    '''
    def variable_weight(self, shape, name, wd=0.01, stddev=0.1):
        var = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
        if wd is not None:
            weight_decay = tf.nn.l2_loss(var) * wd
            tf.add_to_collection('losses', weight_decay)
        return var

    def variable_bias(self, shape, name, constant=0.0):
        return tf.Variable(tf.constant(constant, shape=shape), name)

    def op_conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def op_max_pool_2x2(self, x, use=False):
        if use:
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        else:
            return x

    def op_normalize(self, x, use=False, name='unnamed'):
        if use:
            return tf.nn.lrn(x, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
        else:
            return x

    def op_dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    '''
    训练步骤函数
    '''
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
            kernel = self.variable_weight([self.kernel_size, self.kernel_size, self.layer_depth, self.input_depth], "weights")
            bias = self.variable_bias([self.input_depth], "biases")
            conv = self.op_conv2d(drop, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)

            self.conv_list.append(pre_activation)

        self.net_output_calc = pre_activation

    def evaluation(self):
        self.error = tf.nn.l2_loss(tf.subtract(self.net_output_calc, self.net_output_holder))

    def loss(self):
        l = tf.nn.l2_loss(tf.subtract(self.net_output_calc, self.net_output_holder))
        tf.add_to_collection('losses', l)
        self.l2_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

