# DeepLabCut - XMALab Workflow Instructions

This workflow has been tested on machines running Windows 10 using the DeepLabCut Anaconda Environment

## Step 1: Install XMALab, DeepLabCut, and XROMMTools for DeepLabCut: 

1. Install the latest version of XMALab found [here](https://bitbucket.org/xromm/xmalab/).
2. Follow [DLC documentation](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md) to install Anaconda environment. We recommend using the provided [easy install](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md).
3. If you haven't already, clone or download this repository, and locate the XROMM_DLCTools [functions](/functions/xrommtools.py) and [Demo Jupyter Notebook](/templates/XROMM_Pipeline_Demo.ipynb)
4. **Copy** the xrommtools.py file into the DLC Anaconda environment's utils folder, which can be found where Anaconda was installed- here: ...\Anaconda3\envs\dlc-windowsGPU\Lib\site-packages\deeplabcut\utils

      Or on a Mac:

      .../opt/anaconda3/envs/dlc-macOS-CPU/lib/python3.6/site-packages/deeplabcut/utils

![](https://user-images.githubusercontent.com/53494838/74692595-9ccb9080-51ad-11ea-9906-e6b841238ad7.png)

5.  **Importantly,** Follow DLC documentation to verify successful installation before proceeding.

## Step 2: Track training data in XMALab

_**It is essential**_ that the tracking in this training dataset is as accurate as possible, and that if a marker is not visible, you delete the tracked data for that marker for that frame. Network performance is limited by the quality of the training data.

1. *Track in XMALab*. We recommend tracking 200-600 frames across multiple trials for a single individual. Be sure you are tracking regions that include different poses, as the goal is to capture the variation in the dataset. If you have already tracked several trials, you can proceed to the next step. Having more than 600 frames is fine, as you can decide the number of frames to use for your training dataset.



2. *Export 2D points*. The function xma_to_dlc requires a specific folder struture to bring XMALab tracked data into DeepLabCut. Each trial with frames you wish to include in the training dataset must have its own folder, which must contain the exported (distorted) 2D points csv (use default export settings), as well as the trial images. Trial images can be in the form of AVI files, or cam1/cam2 subfolders with JPG stacks.
- Example folder structure:
  - .../trainingdata/
    - /trial01/
      - /trial01_2dpts.csv 
      - /trial01_camera1.avi
      - /trial01_camera2.avi
    - /trial02/
      - /trial02_2dpts.csv 
      - /trial02_camera1.avi
      - /trial02_camera2.avi
- [Example training data](/templates/trainingdata/) / folder structure for xma_to_dlc
## From this point on, follow along with the provided [Demo Jupyter Notebook](/templates/XROMM_Pipeline_Demo.ipynb). Brief explanations of the steps are are below:

Note that if you wish to create a project with individual networks for each camera plane, use the 2Cam demo notebook.

## Step 3: Create DeepLabCut Project and edit metadata

Follow along with the standard DLC workflow, creating a project and editing the config file to reflect the specifics of your data (i.e. how many points, etc.) 

## Step 4: Create training dataset and train first iteration of neural network

This involves the functions xma_to_dlc, create_training_dataset, and train_network.
- xma_to_dlc requires a folder with subfolders for each trial you tracked frames from. Within each trial subfolder, there should be the 2D points file (from XMALab) and the cam1 and cam2 videos (or folders with JPG stacks).

At this point, be sure to go into the network specific configuration file/files and edit parameters to optimize network performance. Specifically, tuning pose_dist_threshold for marker size, as well as global threshold for scaling.

## Step 5: Analyze (predict points for) new videos

This involves the function analyze_xromm_videos, which calls the native analyze_videos function. You can manually turn on DLC's built-in filter if you wish to filter the predictions before importing into XMALab (the default is for this to be off.)

## Step 6: Import predicted points into XMALab and evaluate performance

In this step, you import predicted 2D points back into XMALab, and assess the quality of the tracking.

## Step 7 (optional): Correct high-error frames, and add to training dataset 

If tracking performance is poor, you can correct select frames and add the to the training dataset, to repeat steps 4-6.
The function add_frames accomplishes this. Add_frames requires a csv file that contains an index of the frames you have corrected. The format of this file is: Column 1 = trialnames, Columns 2:n = frame numbers of corrected frames for those trials. See "example_corrected_frames_idx.csv" for exact formatting.


