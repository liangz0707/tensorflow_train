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

class SRTrainer2(CNNTrainer):
    """
    patch size = 41
    cnn depth = 20
    """
    def __init__(self, model_save_file="", model_load_file="", model_tag=0):
        CNNTrainer.__init__(self,model_save_file=model_save_file, model_load_file=model_load_file, model_tag=model_tag)


if __name__ == "__main__":
    data_file_name = "E:/mySuperResolution/dataset/291/291_cnn_Y_channel_41.pic"
    training_data = load_training_data(data_file_name)
    input = training_data[0][1:400]
    output = training_data[1][1:400]
    # for i in range(100):
    #     plt.subplot(131)
    #     plt.imshow(input[i])
    #     plt.subplot(132)
    #     plt.imshow(output[i])
    #     plt.subplot(133)
    #     plt.imshow(output[i] + input[i])
    #     plt.show()

    #
    with tf.Graph().as_default():
        tr = SRTrainer(model_save_file="E:/mySuperResolution/dataset/test_291",
                       model_load_file="E:/mySuperResolution/dataset/test_291-99500",
                       model_tag=0)
        tr.init_param(layer_depth=64,layer_num=4, input_size=41)
        tr.set_data(input,output)
        tr.setup_frame()
        tr.training()
        re = np.reshape(np.array(tr.restoring()),(-1,21,21))
        for i in range(re.shape[0]):
            plt.subplot(231)
            plt.imshow(input[i])
            plt.subplot(232)
            plt.imshow(output[i])
            plt.subplot(233)
            plt.imshow(output[i] + input[i])
            plt.subplot(235)
            plt.imshow(re[i] )
            plt.subplot(236)
            plt.imshow(re[i] + input[i])
            plt.show()
        print(re.shape)
