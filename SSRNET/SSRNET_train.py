import pandas as pd
import logging
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from SSRNET_model import SSR_net
from TYY_utils import mk_dir, load_data_npz
import numpy as np
import TYY_callbacks
from TYY_generators import *
from keras.utils import plot_model
from moviepy.editor import *
logging.basicConfig(level=logging.DEBUG)




def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input1", "-i1", type=str, required=True,
                        help="path to input1 database npz file")
    parser.add_argument("--input2", "-i2", type=str, required=True,
                        help="path to input2 database npz file")
    parser.add_argument("--db", type=str, required=True,
                        help="database name")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--netType1", type=int, required=True,
                        help="network type 1")
    parser.add_argument("--netType2", type=int, required=True,
                        help="network type 2")

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    input_path1 = args.input1
    input_path2 = args.input2
    db_name = args.db
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    netType1 = args.netType1
    netType2 = args.netType2

    logging.debug("Loading training data...")
    image1, age1, image_size = load_data_npz(input_path1)
    logging.debug("Loading testing data...")
    image2, age2, image_size = load_data_npz(input_path2)
    

    start_decay_epoch = [30,60]

    optMethod = Adam()

    stage_num = [3,3,3]
    lambda_local = 0.25*(netType1%5)
    lambda_d = 0.25*(netType2%5)

    model = SSR_net(image_size,stage_num, lambda_local, lambda_d)()
    save_name = 'ssrnet_%d_%d_%d_%d_%s_%s' % (stage_num[0],stage_num[1],stage_num[2], image_size, lambda_local, lambda_d)
    model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a':'mae'})

    if db_name == "megaage":
        weight_file = "./pre-trained/wiki/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
        model.load_weights(weight_file)
    
    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir(db_name+"_models")
    mk_dir(db_name+"_models/batch_size_%d/"%(batch_size))
    mk_dir(db_name+"_models/batch_size_%d/"%(batch_size)+save_name)
    mk_dir(db_name+"_checkpoints")
    mk_dir(db_name+"_checkpoints/batch_size_%d/"%(batch_size))
    plot_model(model, to_file=db_name+"_models/batch_size_%d/"%(batch_size)+save_name+"/"+save_name+".png")

    with open(os.path.join(db_name+"_models/batch_size_%d/"%(batch_size)+save_name, save_name+'.json'), "w") as f:
        f.write(model.to_json())

    decaylearningrate = TYY_callbacks.DecayLearningRate(start_decay_epoch)

    callbacks = [ModelCheckpoint(db_name+"_checkpoints/batch_size_%d/"%(batch_size)+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",
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
    model.save_weights(os.path.join(db_name+"_models/batch_size_%d/"%(batch_size)+save_name, save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(db_name+"_models/batch_size_%d/"%(batch_size)+save_name, 'history_'+save_name+'.h5'), "history")


if __name__ == '__main__':
    main()

