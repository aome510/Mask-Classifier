import cv2


def plot_training(history):
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# calculate memory usage to train a keras model with batch size = batch_size
def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p)
                              for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p)
                                  for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * \
        (batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def tf_init():
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    # to log device placement (on which device the operation ran)
    config.log_device_placement = True
    sess = tf.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)


def im_show(img, img_name='img', max_size=1024):

    if max(img.shape[0], img.shape[1]) > max_size:
        scale = max_size / max(img.shape[0], img.shape[1])
        img = cv2.resize(img, None, fx=scale, fy=scale)

    cv2.imshow(img_name, img)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


def SSH_init():
    import caffe
    from utils.get_config import cfg, cfg_print, cfg_from_file

    cfg_from_file('./lib/SSH/SSH/configs/wider_pyramid.yml')
    cfg_print(cfg)

    caffe.set_mode_gpu()
    caffe.set_device(0)

    # loading network
    net = caffe.Net('./lib/SSH/SSH/models/test_ssh.prototxt',
                    './lib/SSH/data/SSH_models/SSH.caffemodel', caffe.TEST
                    )
    net.name = 'SSH'

    return net
