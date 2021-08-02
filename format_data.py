"""
Read from nii data and given under-samping scheme and save .mat for network use
Check /utils/data_utils.py for the defined functions for data formatting process

Usage:
1. To generate voxel-wise training dataset for the first N volumes of a full dataset or from a scheme file:
    
    python FormatData.py --path $DataDir --subjects S1 --nDWI N --fc1d_train 
  
    python FormatData.py --path $DataDir --subjects S1 --scheme scheme1 --fc1d_train 

2. For testing dataset, there is no need to reshape the input data to 1-D voxel by voxel.
    Instead, testing on image space could accelerate the computation.
    
    python FormatData.py --path $DataDir --subjects S2 S3 --scheme scheme1 --test 

      (add --Nolabel option if the testing dataset doesn't contain labels)

"""
import argparse
import numpy as np

from utils import gen_dMRI_test_datasets, gen_dMRI_fc1d_train_datasets


parser = argparse.ArgumentParser()
parser.add_argument("--path", help="The path of data folder")
parser.add_argument("--subjects", help="subjects ID", nargs='*')
parser.add_argument("--nDWI", help="The number of volumes", type=int)
parser.add_argument("--scheme", help="The sampling scheme used")
parser.add_argument("--fc1d_train", help="generate fc1d data for training", action="store_true")
parser.add_argument("--test", help="generate base data for testing", action="store_true")
parser.add_argument("--Nolabel", help="generate data without labels for testing only", action="store_true")

args = parser.parse_args()

path = args.path
subjects = args.subjects
fc1d_train = args.fc1d_train
test = args.test
Nolabel = args.Nolabel

# determin the input volumes using the first n volumes
nDWI = args.nDWI
scheme = "first"

# determin the input volumes using a scheme file
combine = None
schemefile = args.scheme
if schemefile is not None:
    combine = np.loadtxt('schemes/' + schemefile)
    combine = combine.astype(int)
    nDWI = combine.sum()
    scheme = schemefile

if test:
    for subject in subjects:
        if Nolabel:
            gen_dMRI_test_datasets(path, subject, nDWI, scheme, combine, fdata=True, flabel=False, whiten=True)
        else: 
            gen_dMRI_test_datasets(path, subject, nDWI, scheme, combine, fdata=True, flabel=True, whiten=True)
if fc1d_train:
    for subject in subjects:
        gen_dMRI_fc1d_train_datasets(path, subject, nDWI, scheme, combine, whiten=True)

