import pandas as pd
import logging
import argparse
import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from TYY_model import TYY_MobileNet_reg, TYY_DenseNet_reg
from TYY_utils import mk_dir, load_data_npz
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
import TYY_callbacks
from keras.preprocessing.image import ImageDataGenerator
'''from mixup_generator import MixupGenerator'''
'''from random_eraser import get_random_eraser'''
from TYY_generators import *
from keras.utils import plot_model
from moviepy.editor import *

logging.basicConfig(level=logging.DEBUG)



def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input1", "-i1", type=str, required=True,
                        help="path to input database npz file")
    parser.add_argument("--input2", "-i2", type=str, required=True,
                        help="path to input database npz file")
    parser.add_argument("--db", type=str, required=True,
                        help="database name")
    parser.add_argument("--netType", type=int, required=True,
                        help="network type")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    input_path1 = args.input1
    input_path2 = args.input2
    db_name = args.db
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    netType = args.netType

    logging.debug("Loading training data...")
    image1, age1, image_size = load_data_npz(input_path1)
    logging.debug("Loading testing data...")
    image2, age2, image_size = load_data_npz(input_path2)
    

    start_decay_epoch = [30,60]

    optMethod = Adam()

    if netType == 1:
        model_type = 'MobileNet'
        alpha = 0.25
        model = TYY_MobileNet_reg(image_size,alpha)()
        save_name = 'mobilenet_reg_%s_%d' % (alpha, image_size)
        model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a':'mae'})

    elif netType == 2:
        model_type = 'MobileNet'
        alpha = 0.5
        model = TYY_MobileNet_reg(image_size,alpha)()
        save_name = 'mobilenet_reg_%s_%d' % (alpha, image_size)
        model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a':'mae'})

    elif netType == 3:
        model_type = 'DenseNet'
        N_densenet = 3
        depth_densenet = 3*N_densenet+4
        model = TYY_DenseNet_reg(image_size,depth_densenet)()
        save_name = 'densenet_reg_%d_%d' % (depth_densenet, image_size)
        model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a':'mae'})

    elif netType == 4:
        model_type = 'DenseNet'
        N_densenet = 5
        depth_densenet = 3*N_densenet+4
        model = TYY_DenseNet_reg(image_size,depth_densenet)()
        save_name = 'densenet_reg_%d_%d' % (depth_densenet, image_size)
        model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a':'mae'})

    

    if db_name == "meagaage":
        weight_file = "../pre-trained/wiki/"+save_name+"/"+save_name+".h5"
        model.load_weights(weight_file)

    
    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")

    mk_dir(db_name+"_models")
    mk_dir(db_name+"_models/"+model_type+"/")
    mk_dir(db_name+"_models/"+model_type+"/batch_size_%d/"%(batch_size))
    mk_dir(db_name+"_models/"+model_type+"/batch_size_%d/"%(batch_size)+save_name)
    mk_dir(db_name+"_checkpoints")
    mk_dir(db_name+"_checkpoints/"+model_type)
    mk_dir(db_name+"_checkpoints/"+model_type+"/batch_size_%d/"%(batch_size))
    plot_model(model, to_file=db_name+"_models/"+model_type+"/batch_size_%d/"%(batch_size)+save_name+"/"+save_name+".png")

    with open(os.path.join(db_name+"_models/"+model_type+"/batch_size_%d/"%(batch_size)+save_name, save_name+'.json'), "w") as f:
        f.write(model.to_json())

    decaylearningrate = TYY_callbacks.DecayLearningRate(start_decay_epoch)

    callbacks = [ModelCheckpoint(db_name+"_checkpoints/"+model_type+"/batch_size_%d/"%(batch_size)+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"), decaylearningrate
                        ]

    logging.debug("Running training...")
    

    data_num = len(image1)+len(image2)
    indexes1 = np.arange(len(image1))
    indexes2 = np.arange(len(image2))
    np.random.shuffle(indexes1)
    np.random.shuffle(indexes2)
    x_train = image1[indexes1]
    x_test = image2[indexes2]
    y_train_a = age1[indexes1]
    y_test_a = age2[indexes2]
    train_num = len(image1)
    
    
    hist = model.fit_generator(generator=data_generator_reg(X=x_train, Y=y_train_a, batch_size=batch_size),
                               steps_per_epoch=train_num // batch_size,
                               validation_data=(x_test, [y_test_a]),
                               epochs=nb_epochs, verbose=1,
                               callbacks=callbacks)

    logging.debug("Saving weights...")
    model.save_weights(os.path.join(db_name+"_models/"+model_type+"/batch_size_%d/"%(batch_size)+save_name, save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(db_name+"_models/"+model_type+"/batch_size_%d/"%(batch_size)+save_name, 'history_'+save_name+'.h5'), "history")


if __name__ == '__main__':
    main()
