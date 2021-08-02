# Acceleration-of-microstructure-imaging-in-diffusion-MRI

This repository contains the code for training a neural network model that reconstruct diffusion tensor imaging (DTI) from highly accelerated scans. Based on insipiration from the following paper:

[Highly accelerated, model-free diffusion tensor MRI reconstrucion using neural networks](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.13400?af=R), we reconstructed Fractional Anisotropy (FA) and Mean Diffusivity (MD) maps from two subjects in [multi-shell diffusion MRI dataset](https://www.nature.com/articles/s41597-020-0493-8?sf234363855=1).

## Usage

#### Step 1: Choose the under-sampled image volumes from full diffusion dataset (the nifty files), and format and save training and testing data to .mat file; including some pre-processing steps

    # a. Formatting training dataset with the first 10 image volumes from the full diffusion dataset
        python format_data.py --path /Your/Data/Dir --subjects S1 --nDWI 10 --fc1d_train
  
        # You can also format training dataset with a scheme file contained in the ${NetDir}/schemes folder, 
        # which are 1 for the target image volumes to choose and 0 for all other volumes. (see example file scheme1)
        # Use when you want to design new under-sampling schemes
        python format_data.py --path /Your/Data/Dir/Data-DTI --subjects S1 --scheme scheme1 --fc1d_train 
        
        
    # b. Formatting test dataset (add --Nolabel option if the dataset contains no available labels
        python format_data.py --path $DataDir --subjects S2 --nDWI 10 --test
        
      
 #### Step 2: Network training; Check all available options and default values in /utils/model.py

    # Using the first 10 volumes; you can also use a scheme file to determined the input DWI volumes. 
        python training.py --train_subjects S1 --DWI 10 --model fc1d --train 



##### Step 3: Test the model with dataset from S2; weights are saved from previous training
        python testing.py --test_subject S2 --DWI 10 --model fc1d
        
## Sample results
        

