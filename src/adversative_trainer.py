# coding:utf-8
from trainer import *
from data_utils import *
import sys

class AdversativeTrainer(CNNTrainer):

    def inference_D(self, x , z):
        input_x = tf.reshape(x, [-1, 21 * 21 * 1])
        input_z = tf.reshape(z, [-1, 21 * 21 * 1])
        var_list = []
        with tf.variable_scope('xh1') as scope:
            W = tf.Variable(tf.random_normal([21 * 21 * 1, 16]))
            b = tf.Variable(tf.random_normal([16]))

            conv_x = tf.matmul(input_x ,W) + b
            layer1_x = tf.nn.relu(conv_x)

            conv_z = tf.matmul(input_z, W) + b
            layer1_z = tf.nn.relu(conv_z)

            var_list.append(W)
            var_list.append(b)

        with tf.variable_scope('xh2') as scope:
            W = tf.Variable(tf.random_normal([16, 1]))
            b = tf.Variable(tf.random_normal([1]))

            conv_x = tf.matmul(layer1_x, W) + b

            conv_z = tf.matmul(layer1_z, W) + b

            var_list.append(W)
            var_list.append(b)

        return conv_x, conv_z, var_list

    def inference_G(self, x):
        var_list = []
        with tf.variable_scope('c1') as scope:
            kernel = self.variable_weight([5, 5, 1, 5], "weights",stddev=0.001)
            bias = self.variable_bias([5], "biases")
            conv = self.op_conv2d(x, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv_out = tf.nn.relu(pre_activation, name=scope.name)
            var_list.append(kernel)
            var_list.append(bias)

        with tf.variable_scope('c2') as scope:
            kernel = self.variable_weight([5, 5, 5, 1], "weights",stddev=0.001)
            bias = self.variable_bias([1], "biases")
            conv = self.op_conv2d(conv_out, kernel)
            pre_activation = tf.nn.bias_add(conv, bias)
            var_list.append(kernel)
            var_list.append(bias)
        return pre_activation,var_list


if __name__ == "__main__":
    image = cv2.imread("E:/mySuperResolution/dataset/Set14/comic[1-Original].bmp")

    img = np.reshape(image[0:21,0:21,0],(1,21,21,1))
    s = img.shape
    rand_img = np.random.random(s)

    a = AdversativeTrainer()
    x_holder = tf.placeholder(tf.float32, shape=(None, s[1], s[2], 1))
    z_holder = tf.placeholder(tf.float32, shape=(None, s[1], s[2], 1))

    y_g, var_list_g = a.inference_G(x_holder)

    y,z,var_list_d = a.inference_D(y_g,z_holder)

    obj_d = tf.reduce_mean(tf.sigmoid(z)+ 1- tf.sigmoid(y))
    opt_d = tf.train.GradientDescentOptimizer(0.001).minimize(-obj_d,var_list=var_list_d)

    obj_g = tf.reduce_mean(tf.sigmoid(y))
    opt_g = tf.train.GradientDescentOptimizer(0.001).minimize(-obj_g,var_list=var_list_g)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):

        _,p1,p2 = sess.run([opt_d,obj_d,obj_g], {x_holder: rand_img, z_holder: img})
        print (p1,p2)
        for j in range(5):
            sess.run(opt_g, {x_holder: rand_img})
    print(sess.run(y_g, {x_holder: rand_img}))
    print ("---")
    print (img)