"""
Module definition for network training and testing.

Define your new model here

"""

import os
import argparse
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Conv3D, Flatten, Reshape, Conv2DTranspose, UpSampling2D, Concatenate


class MRIModel(object):
    """
    MRI models
    """

    _ndwi = 0
    _single = False
    _model = None
    _type = ''
    _loss = []
    _label = ''
    _kernel1 = 150
    _kernel2 = 50
    _kernel3 = 150

    def __init__(self, ndwi=96, model='fc1d', layer=3, train=True, kernels=None, test_shape=[90, 90, 90]):
        self._ndwi = ndwi
        self._type = model
        self._hist = None
        self._train = train
        self._layer = layer
        self._test_shape = test_shape
        if kernels is not None:
            self._kernel1, self._kernel2, self._kernel3 = kernels
   
    def _fc1d_model(self, patch_size):
        """
        Fully-connected 1d ANN model.
        """
        inputs = Input(shape=(self._ndwi,))
        # Define hidden layer
        hidden = Dense(self._kernel1, activation='relu')(inputs)
        for i in np.arange(self._layer  - 1):
            print(i)
            hidden = Dense(self._kernel1, activation='relu')(hidden)

        hidden = Dropout(0.1)(hidden)

        # Define output layer
        outputs = Dense(2, name='output', activation='relu')(hidden)

        self._model = Model(inputs=inputs, outputs=outputs)




    """
    Define your new model here.

    """
    def _diffnet_model(self,patch_size):

        inputs = Input(shape=(self._ndwi))

        hidden  = Dense(self._kernel1,activation='relu')(inputs)
        hidden = Dense(self._kernel2,activation='relu')(hidden)
        outputs = Dense(2,name='output',activation='tanh')(hidden)

        self._model = Model(inputs=inputs,outputs=outputs)



    
    def _conv3d_model(self, patch_size):
        """
        Conv3D model.
        """
        if self._train:
            inputs = Input(shape=(patch_size, patch_size, patch_size, self._ndwi))
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, dim2, self._ndwi))
        hidden = Conv3D(self._kernel1, 3, activation='relu', padding='valid')(inputs)
        for i in np.arange(self._layer - 1):
            hidden = Conv3D(self._kernel1, 1, activation='relu', padding='valid')(hidden)
        hidden = Dropout(0.1)(hidden)
        outputs = Conv3D(2, 1, activation='relu', padding='valid')(hidden)

        self._model = Model(inputs=inputs, outputs=outputs)

    __model = {
        'fc1d' : _fc1d_model,
        'diffnet':_diffnet_model,
        'conv3d' : _conv3d_model,
    }

    def model(self, optimizer, loss, patch_size):
        """
        Generate model.
        """
        self.__model[self._type](self, patch_size)
        self._model.summary()
        self._model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def _sequence_train(self, data, label, nbatch, epochs, callbacks, shuffle, validation_data):

        validation_split = 0.0
        if validation_data is None:
            validation_split = 0.2

        self._hist = self._model.fit(data, label,
                                     batch_size=nbatch,
                                     epochs=epochs,
                                     shuffle=shuffle,
                                     validation_data=validation_data,
                                     validation_split=validation_split,
                                     callbacks=callbacks)
        self._loss.append(len(self._hist.history['loss']))
        self._loss.append(self._hist.history['loss'][-1])
        self._loss.append(None)
        self._loss.append(self._hist.history['accuracy'][-1])
        self._loss.append(None)

    __train = {
        'fc1d' : _sequence_train,
        'diffnet':_sequence_train,
        'conv3d' : _sequence_train,
    }

    def train(self, data, label, nbatch, epochs, callbacks, weightname,
              shuffle=True, validation_data=None):
        """
        Training on training datasets.
        """
        #print "Training start ..."
        self.__train[self._type](self, data, label, nbatch, epochs,
                                 callbacks, shuffle, validation_data)

        try:
            self._model.save_weights('weights/' + weightname + '.weights')
        except IOError:
            os.system('mkdir weights')
            self._model.save_weights('weights/' + weightname + '.weights')

        return self._loss

    def load_weight(self, weightname):
        """
        Load pre-trained weights.
        """
        self._model.load_weights('weights/' + weightname + '.weights')

    def predict(self, data):
        """
        Predict on test datas.
        """
        pred = self._model.predict(data)
        if self._type[-6:] == 'staged':
            pred = np.concatenate((pred[0], pred[1]), axis=-1)

        return pred

def parser():
    """
    Create a parser.
    """
    parser = argparse.ArgumentParser()
    
    # Specify train & test sets
    parser.add_argument("--train_subjects", help="Training subjects IDs", nargs='*')
    parser.add_argument("--test_subject", help="Testing subject ID", nargs='*')
    parser.add_argument("--scheme", metavar='name', help="The scheme for sampling", default='first')
    parser.add_argument("--DWI", metavar='N', help="Number of input DWI volumes", type=int, default=60)
  
   # Training parameters
    parser.add_argument("--train", help="Train the network", action="store_true")
    parser.add_argument("--model", help="Train model",
                        choices=['fc1d', 'diffnet', 'conv3d'], default='diffnet')
    parser.add_argument("--layer", metavar='l', help="Number of layers", type=int, default=3)
    parser.add_argument("--lr", metavar='lr', help="Learning rates", type=float, default=0.001)
    parser.add_argument("--epoch", metavar='ep', help="Number of epoches", type=int, default=100)
    parser.add_argument("--kernels", help="The number of kernels for each layer", nargs='*',
                        type=int, default=None)
        
    # Just For test; not use anymore
    parser.add_argument("--loss", help="Set different loss functions", action="store_true")
    parser.add_argument("--test_shape", nargs='*', type=int, default=None)
    parser.add_argument("--batch", metavar='bn', help="Batch size", type=int, default=256)
    parser.add_argument("--patch_size", metavar='ksize', help="Size of the kernels", type=int, default=3)
    parser.add_argument("--base", metavar='base', help="choice of training data", type=int, default=1)    

    return parser
