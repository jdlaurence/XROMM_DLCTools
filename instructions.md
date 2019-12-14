# DeepLabCut - XMALab Workflow Instructions

This workflow has been tested on machines running Windows 10 using the DeepLabCut Anaconda Environment

## Step 1: Install XMALab, DeepLabCut, and XROMMTools for DeepLabCut: 

1. Install the latest version of XMALab found [here](https://bitbucket.org/xromm/xmalab/).
2. Follow [DLC documentation](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md) to install Anaconda environment. We recommend using the provided [easy install](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md).
3. If you haven't already, clone or download this repository, and locate the XROMMTools for DeepLabCut [functions](xrommtools.py) and [Demo Jupyter Notebook](XROMM_Pipeline_Demo.ipynb)
4. **Copy** the xrommtools.py file into the DLC Anaconda environment's utils folder, which can be found where Anaconda was installed- here: ...\Anaconda3\envs\dlc-windowsGPU\Lib\site-packages\deeplabcut\utils
5. Follow DLC documentation to verify successful installation before proceeding.

## Step 2: Track training data in XMALab

_**It is essential**_ that the tracking in this training dataset is as accurate as possible, and that if a marker is not visible, you delete the tracked data for that marker for that frame. Network performance is limited by the quality of the training data.

1. *Track in XMALab*. We recommend tracking 200-600 frames across multiple trials for a single individual. Be sure you are tracking regions that include different poses, as the goal is to capture the variation in the dataset. If you have already tracked several trials, you can proceed to the next step. Having more than 600 frames is fine, as you can decide the number of frames to use for your training dataset.



2. *Export 2D points*. The XROMMTools function xma_to_dlc requires a specific folder struture to bring XMALab tracked data into DeepLabCut. Each trial with frames you wish to include in the training dataset must have its own folder, which must contain the export 2D points csv, as well as the trial images. See image below.  

## From this point on, follow along with the provided [Demo Jupyter Notebook](XROMM_Pipeline_Demo.ipynb). Brief explanations of the steps are are below:

## Step 3: Create DeepLabCut Project and edit metadata

## Step 4: Create training dataset and train first iteration of neural network

## Step 5: Analyze (predict points for) new videos

## Step 6: Import predicted points into XMALab and evaluate performance

## Step 7 (optional): Correct high-error frames, and add to training dataset 



