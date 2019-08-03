from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.callbacks import ModelCheckpoint
from keras.applications import resnet50, inception_resnet_v2
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from func import tf_init, plot_training, get_model_memory_usage
import os


def train(type):  # we deploy two networks resnet50 and inceptionresnet_v2
    os.system('mkdir ./model/')
    os.system('mkdir ./model/snapshot/')
    os.system('mkdir ./model/output/')

    if type == 'resnet50':
        csize = 224
        batch_size = 32
        preprocess_input = resnet50.preprocess_input
        base_model = ResNet50(
            weights='imagenet', pooling='avg', include_top=False, input_shape=(csize, csize, 3))
    else:
        csize = 299
        batch_size = 16
        preprocess_input = inception_resnet_v2.preprocess_input
        base_model = InceptionResNetV2(
            weights='imagenet', pooling='avg', include_top=False, input_shape=(csize, csize, 3))

    predictions = Dense(1, activation='sigmoid')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)
    # summary network
    # model.summary()

    # calculate memory used for training network
    # print(get_model_memory_usage(batch_size, model))

    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(sgd, loss='binary_crossentropy', metrics=['accuracy'])

    data_gen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        preprocessing_function=preprocess_input
    )

    train_gen = data_gen.flow_from_directory(
        directory='./data/train/',
        class_mode='binary',
        batch_size=batch_size,
        target_size=(csize, csize)
    )
    val_gen = data_gen.flow_from_directory(
        directory='./data/test/',
        class_mode='binary',
        batch_size=batch_size,
        target_size=(csize, csize)
    )

    model_saver = ModelCheckpoint(
        './model/snapshot/' + type +
        '-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.h5',
        save_best_only=True, monitor='val_acc')
    callbacks = [model_saver]

    history = model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=callbacks,
        steps_per_epoch=int(train_gen.n / train_gen.batch_size),
        validation_steps=int(val_gen.n / val_gen.batch_size)
    )

    plot_training(history)

    model.save('./model/output/{}.h5'.format(type))


if __name__ == "__main__":
    tf_init()
    train('resnet50')
