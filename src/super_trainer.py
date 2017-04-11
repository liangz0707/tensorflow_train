# coding:utf-8
import matplotlib.pyplot as plt
from trainer import *
from data_utils import *
import configparser
import sys
import super_restore

class SRTrainer(CNNTrainer):

    def __init__(self, model_save_file="", model_load_file="", model_tag=0):
        CNNTrainer.__init__(self,model_save_file=model_save_file, model_load_file=model_load_file, model_tag=model_tag)


class SRTrainerDeconDeep(CNNTrainer):

    def __init__(self, model_save_file="", model_load_file="", model_tag=0):
        CNNTrainer.__init__(self,model_save_file=model_save_file, model_load_file=model_load_file, model_tag=model_tag)

    def op_conv2d_trans(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape,  strides = [1, 1, 1, 1], padding='SAME')

    def init_param(self, layer_depth=32, layer_num=8, input_size=21, input_depth=1 ,
                  kernel_size=3, keep_prob_value = 1.0, USE_POOL = False, USE_NORM = False,
                   reload = False,batch_size = 64,itr_num=200000,save_step=500,con_num=9,decon_num=0, test_dir="",test_result_pefix="",
                   learning_rate=0.0001):
        CNNTrainer.init_param(self, layer_depth=layer_depth, layer_num=layer_num,input_size=input_size,input_depth=input_depth,
                              kernel_size=kernel_size,  keep_prob_value=keep_prob_value, USE_POOL=USE_POOL,
                              reload=reload,batch_size=batch_size, itr_num=itr_num, save_step=save_step,
                              test_dir=test_dir,test_result_pefix=test_result_pefix,learning_rate=learning_rate)
        self.con_num = con_num
        self.decon_num = decon_num

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

        for i in range(int(self.con_num)):
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

        for i in range(int(self.decon_num)):
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

            self.conv_list.append(pre_activation)
        self.net_output_calc = pre_activation


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_tag = "db"
    else:
        run_tag = sys.argv[1]
        print ("训练数据标签：%s" & sys.argv[1])

    cf = configparser.ConfigParser()
    cf.read("trainer.conf")
    layer_depth = cf.getint(run_tag, "layer_depth")
    layer_num = cf.getint(run_tag, "layer_num")
    batch_size = cf.getint(run_tag, "batch_size")
    keep_prob_value = cf.getfloat(run_tag, "keep_prob_value")
    con_num = cf.getint(run_tag, "con_num")
    decon_num = cf.getint(run_tag, "decon_num")
    input_size = cf.getint(run_tag, "input_size")
    test_dir = cf.get(run_tag, "test_dir")
    model_file = cf.get(run_tag, "model_file")
    model_save_file = cf.get(run_tag, "model_save_file")
    model_tag = cf.getint(run_tag, "model_tag")
    train_data_file =  cf.get(run_tag, "train_data_file")
    itr_num =  cf.getint(run_tag, "itr_num")
    test_result_pefix = cf.get(run_tag, "test_result_pefix")
    learning_rate = cf.getfloat(run_tag, "learning_rate")
    save_step = cf.getint(run_tag, "save_step")

    data_file_name = train_data_file
    training_data = load_training_data(data_file_name)
    print ("训练数据共有%d" % len(training_data[0]))
    input = training_data[0]
    output = training_data[1]

    with tf.Graph().as_default():
        tr = SRTrainerDeconDeep(model_save_file=model_save_file,
                       model_load_file=model_file,
                       model_tag=model_tag)
        tr.init_param(itr_num=itr_num, batch_size=batch_size, layer_depth=layer_depth,
                      layer_num=layer_num, input_size=input_size,keep_prob_value = keep_prob_value,
                      con_num=con_num,decon_num=decon_num, test_dir=test_dir,test_result_pefix=test_result_pefix, learning_rate=learning_rate,
                        save_step=save_step)
        tr.set_data(input,output)
        tr.setup_frame()
        tr.training()
