import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_nii_image


parser = argparse.ArgumentParser()
parser.add_argument("--test_image", help="the path to the test image")
parser.add_argument("--gt_image", help ="the path to the ground truth image")
parser.add_argument("--frame_number", help="frame _number")

args = parser.parse_args()
test_image = args.test_image
gt_image = args.gt_image
frame_num = args.frame_number

if test_image:
    data = load_nii_image(test_image)

    norm= np.linalg.norm(data)
    normed_data = data/norm
    #ploting nii
    img = normed_data[:, :, int(frame_num)]
    plt.axis('off')
    sns.heatmap(img, cmap='gray')
    plt.show()

    #plotting error
    gt_data = load_nii_image(gt_image)
    normalize = np.linalg.norm(gt_data)
    gt_data = gt_data / normalize
    gt_img = gt_data[:,:,int(frame_num)]
    plt.axis('off')
    err = img - gt_img
    sns.heatmap(err, cmap='bwr')
    plt.show()










