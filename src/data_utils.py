import sys
if (sys.version == 2):
    import cPickle
else:
    import _pickle as cPickle

def load_training_data(data_file_name):
    training_data = None
    with open(data_file_name, 'rb') as f:
        training_data = cPickle.load(f, encoding='iso-8859-1')
    #
    # with open("291_cnn_Y_channel.pic", 'rb') as f:
    #     training_data = cPickle.load(f)
    return training_data

