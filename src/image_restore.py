# coding:utf-8
from super_trainer import *
from super_restore import *

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("trainer.conf")
    layer_depth = cf.getint("db", "layer_depth")
    layer_num = cf.getint("db", "layer_num")
    batch_size = cf.getint("db", "batch_size")
    keep_prob_value = cf.getfloat("db", "keep_prob_value")
    con_num = cf.getint("db", "con_num")
    decon_num = cf.getint("db", "decon_num")
    input_size = cf.getint("db", "input_size")
    test_dir = cf.get("db", "test_dir")
    model_file = cf.get("db", "model_file")
    model_save_file = cf.get("db", "model_save_file")
    model_tag =  cf.getint("db", "model_tag")
    reload = cf.getboolean("db","reload")
    test_result_pefix = cf.get("db", "test_result_pefix")


    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            tr = SRTrainerDeconDeep(model_save_file=model_save_file,
                                    model_load_file=model_file,
                                    model_tag=model_tag)
            tr.init_param(layer_depth=layer_depth, layer_num=layer_num, input_size=input_size, batch_size=batch_size,
                          keep_prob_value=keep_prob_value,con_num=con_num, decon_num=decon_num,
                          reload = reload,test_result_pefix=test_result_pefix)
            tr.setup_frame()
            tr.sess.run(tf.global_variables_initializer())
            tr.saver = tf.train.Saver()
            tr.saver.restore(tr.sess, tr.model_load_file)

            restore_dir(test_dir,tr,test_result_pefix)
