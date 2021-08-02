"""
Main script for network training and testing
Definition of the command-line arguments are in model.py and can be displayed by `python Testing.py -h`

"""

import numpy as np
import os
import time

from scipy.io import savemat, loadmat

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                                            EarlyStopping

from utils import save_nii_image, calc_RMSE, loss_func, repack_pred_label, \
                  MRIModel, parser, load_nii_image, unmask_nii_data, loss_funcs, fetch_train_data_MultiSubject


# Get parameter from command-line input
args = parser().parse_args()

train_subjects = args.train_subjects
test_subject = args.test_subject[0]
nDWI = args.DWI
scheme = args.scheme
mtype = args.model

lr = args.lr
epochs = args.epoch
kernels = args.kernels
layer = args.layer

loss = args.loss
batch_size = args.batch
patch_size = args.patch_size
label_size = patch_size - 2
base = args.base

# Constants
types = ['FA' , 'MD']
ntypes = len(types)
decay = 0.1

# Parameter name definition
savename = str(nDWI)+ '-'  + scheme + '-' + args.model

# Define the adam optimizer
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Load testing data
mask = load_nii_image('datasets/mask/mask_' + test_subject + '.nii')
tdata = loadmat('datasets/data/' + test_subject + '-' + str(nDWI) + '-' + scheme + '.mat')['data']

test_shape = args.test_shape
if test_shape is None:
  test_shape = tdata.shape[1:4]

# Define the model
model = MRIModel(nDWI, model=mtype, layer=layer, train=False, kernels=kernels, test_shape=test_shape)
model.model(adam, loss_func, patch_size)
model.load_weight(savename)

weights = model._model.layers[1].get_weights()

# Predict on the test data.
time1 = time.time()
pred = model.predict(tdata)
time2 = time.time()

time3 = time.time()
pred = repack_pred_label(pred, mask, mtype, ntypes)
time4 = time.time()

#print "predict done", time2 - time1, time4 - time3

# Save estimated measures to /nii folder as nii image
os.system("mkdir -p nii")

for i in range(ntypes):
    data = pred[..., i]
    filename = 'nii/' + test_subject + '-' + types[i] + '-' + savename + '.nii'

    data[mask == 0] = 0
    save_nii_image(filename, data, 'datasets/mask/mask_' + test_subject + '.nii', None)
