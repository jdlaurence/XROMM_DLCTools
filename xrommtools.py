"""
XROMM Tools for DeepLabCut
Developed by J.D. Laurence-Chasen

Functions:

xma_to_dlc: create DeepLabCut training dataset from data tracked in XMALab
analyze_xromm_videos: Predict 2D points for novel trials
dlc_to_xma: convert output of DeepLabCut to XMALab format 2D points fiel
add_frames: Add new frames corrected/tracked in XMALab to an existing training dataset

"""


import os
import pandas as pd
import numpy as np
import cv2
import random
import deeplabcut

def xma_to_dlc(path_config_file_cam1,path_config_file_cam2,data_path,dataset_name,scorer,nframes):
    configs = [path_config_file_cam1[:-12], path_config_file_cam2[:-12]]
    cameras = [1,2]
    picked_frames = [] 
    dfs = []
    idx = []
    pnames = []
    subs =[["c01","c1","C01","C1","Cam1","cam1","Cam01","cam01","Camera1","camera1"],["c02","c2","C02","C2","Cam2","cam2","Cam02","cam02","Camera2","camera2"]]
    trialnames = os.listdir(data_path)


    ### PART 1: Pick frames for dataset

    for trial in trialnames:

        # Read 2D points file
        contents = os.listdir(data_path+"/"+trial)
        filename = [x for x in contents if ".csv" in x] # csv filename
        df1=pd.read_csv(data_path+"/"+trial+"/"+filename[0], sep=',',header=None)

        # read pointnames from header row
        pointnames = df1.loc[0,::4].astype(str).tolist()
        # get rid of leading or trailing zeros
        for point in range(len(pointnames)):
            pointnames[point] = pointnames[point].strip()
            # get rid of cam1/cam2/x/y
            pointnames[point] = pointnames[point][:-7]
        pnames.append(pointnames)

        df1 = df1.loc[1:,].reset_index(drop=True) # remove header row

        # temp_idx = rows where fewer than half of columns are ~NaN
        ncol = df1.shape[1]
        temp_idx = list(df1.index.values[(~pd.isnull(df1)).sum(axis = 1) >= ncol/2])


        # randomize frames w/i each trial and append to index list
        random.shuffle(temp_idx)
        idx.append(temp_idx)
        dfs.append(df1)


    #if pointnames aren't the same across trials    
    #if any(pnames[0] != x for x in pnames):
        #print(pnames)
       # raise ValueError('Make sure point names are consistent across trials')

    
    if nframes == 'all':
        
        picked_frames = idx # use all detected frames
        
    else:
        # pick frames to extract (NOTE this is random currently)
        # current code iteratively picks one frame at a time from each shuffled trial until # of picked_frames hits nframes
        # There is a much neater way to do this
        if sum(len(x) for x in idx) < nframes:
            raise ValueError('nframes is bigger than number of detected frames')

        count = 0
        while sum(len(x) for x in picked_frames) < nframes:
            for trialnum in range(len(idx)):
                if sum(len(x) for x in picked_frames) < nframes:
                    if count == 0:
                        picked_frames.insert(trialnum,[idx[trialnum][count]])
                    elif count <= len(idx[trialnum]):
                        picked_frames[trialnum] = picked_frames[trialnum]+[idx[trialnum][count]]
            count += 1

    ### Part 2: Extract images and 2D point data

    for camera in cameras:
        print("Extracting camera %d trial images and 2D points..."%camera)
        relnames = []
        data = pd.DataFrame()
        # new training dataset folder  
        newpath = configs[camera-1]+"/labeled-data/"+dataset_name+"_cam"+str(camera)
        h5_save_path = newpath+"/CollectedData_"+scorer+".h5"
        csv_save_path = newpath+"/CollectedData_"+scorer+".csv"

        if os.path.exists(newpath):
            contents = os.listdir(newpath)
            if contents:
                raise ValueError('There are already data in the camera %d training dataset folder' %camera)
        else:
            os.makedirs(newpath) # make new folder

        for trialnum,trial in enumerate(trialnames):

            # get video file
            ### IMPORTANT. 
            file = []
            contents = os.listdir(data_path+"/"+trial)
            for name in contents:
                if any(x in name for x in subs[camera-1]):
                    file = name
            if not file:
                raise ValueError('Cannot locate %s video file or image folder' %trial)

            # if video file is actually folder of frames
            if os.path.isdir(data_path+"/"+trial+"/"+file):
                imgpath = data_path+"/"+trial+"/"+file
                imgs = os.listdir(imgpath)
                relpath = "labeled-data/"+dataset_name+"_cam"+str(camera)+"/"
                frames = picked_frames[trialnum]
                frames.sort()
                count2 = 0
                for count,img in enumerate(imgs):
                    if count in frames:
                        image = cv2.imread(imgpath+"/"+img)
                        relname = relpath + trial + "_%s.png" % str(count+1).zfill(4)
                        relnames = relnames + [relname]
                        cv2.imwrite(newpath + "/" + trial + "_%s.png" % str(count+1).zfill(4), image) # save frame
                        count2 += 1


            else:
            # file is actually a file        
            # extract frames from video and convert to png
                video = data_path+"/"+trial+"/"+file
                relpath = "labeled-data/"+dataset_name+"_cam"+str(camera)+"/"
                frames = picked_frames[trialnum]
                frames.sort()
                cap = cv2.VideoCapture(video)
                success,image = cap.read()
                count = 0
                count2 = 0
                while success:
                    if count in frames:
                        relname = relpath + trial + "_%s.png" % str(count+1).zfill(4)
                        relnames = relnames + [relname]
                        cv2.imwrite(newpath + "/" + trial + "_%s.png" % str(count+1).zfill(4), image) # save frame
                        count2 += 1
                    success,image = cap.read()
                    count += 1
                cap.release()
                
            
            if count2 != len(frames):
                raise ValueError('problem with %s. Frame counts don"t match. Make sure trial folders only contain video files and 2D points file.'%trial)
            print("%d frames from "%len(frames) + trial + " extracted" )       
            
            # extract 2D points data
            df1 = dfs[trialnum]
            xpos = df1.iloc[frames,0+(camera-1)*2::4]
            ypos = df1.iloc[frames,1+(camera-1)*2::4]
            temp_data = pd.concat([xpos,ypos],axis=1).sort_index(axis=1)
            data = pd.concat([data,temp_data])

    ### Part 3: Complete final structure of datafiles
        dataFrame = pd.DataFrame()
        temp = np.empty((data.shape[0],2,))
        temp[:] = np.nan
        for i,bodypart in enumerate(pointnames):
            index = pd.MultiIndex.from_product([[scorer], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
            frame = pd.DataFrame(temp, columns = index, index = relnames)
            frame.iloc[:,0:2] = data.iloc[:, 2*i:2*i+2].values.astype(float)
            dataFrame = pd.concat([dataFrame, frame],axis=1)
        dataFrame.replace('', np.nan, inplace=True)
        dataFrame.to_hdf(h5_save_path, key="df_with_missing", mode="w")
        dataFrame.to_csv(csv_save_path,na_rep='NaN')
        print("...done.")

    print("Training data extracted to projectpath/labeled-data. Now use deeplabcut.create_training_dataset")

def dlc_to_xma(cam1data,cam2data,trialname,savepath):
    
    h5_save_path = savepath+"/"+trialname+"-Predicted2DPoints.h5"
    csv_save_path = savepath+"/"+trialname+"-Predicted2DPoints.csv"
    
    if isinstance(cam1data, str): #is string
        if ".csv" in cam1data:

            cam1data=pd.read_csv(cam1data, sep=',',header=None)
            cam2data=pd.read_csv(cam2data, sep=',',header=None)
            pointnames = list(cam1data.loc[1,1:].unique())
            
            # reformat CSV / get rid of headers
            cam1data = cam1data.loc[3:,1:]
            cam1data.columns = range(cam1data.shape[1])
            cam1data.index = range(cam1data.shape[0])
            cam2data = cam2data.loc[3:,1:]
            cam2data.columns = range(cam2data.shape[1])
            cam2data.index = range(cam2data.shape[0])
            
        elif ".h5" in cam1data:# is .h5 file
            cam1data = pd.read_hdf(cam1data)
            cam2data = pd.read_hdf(cam2data)
            pointnames = list(cam1data.columns.get_level_values('bodyparts').unique())

        else:
            raise ValueError('2D point input is not in correct format')
    else:
        
        pointnames = list(cam1data.columns.get_level_values('bodyparts').unique())
    
    # make new column names
    nvar = len(pointnames)
    pointnames = [item for item in pointnames for repetitions in range(4)]
    post = ["_cam1_X", "_cam1_Y", "_cam2_X", "_cam2_Y"]*nvar
    cols = [m+str(n) for m,n in zip(pointnames,post)]


    # remove likelihood columns
    cam1data = cam1data.drop(cam1data.columns[2::3],axis=1)
    cam2data = cam2data.drop(cam2data.columns[2::3],axis=1)

    # replace col names with new indices
    c1cols = list(range(0,cam1data.shape[1]*2,4)) + list(range(1,cam1data.shape[1]*2,4))
    c2cols = list(range(2,cam1data.shape[1]*2,4)) + list(range(3,cam1data.shape[1]*2,4))
    c1cols.sort()
    c2cols.sort()
    cam1data.columns = c1cols
    cam2data.columns = c2cols

    df = pd.concat([cam1data,cam2data],axis=1).sort_index(axis=1)
    df.columns = cols
    df.to_hdf(h5_save_path, key="df_with_missing", mode="w")
    df.to_csv(csv_save_path,na_rep='NaN',index=False)

def analyze_xromm_videos(path_config_file_cam1,path_config_file_cam2,new_data_path,iteration):
# assumes you have cam1 and cam2 videos as .avi in their own seperate trial folders
# assumes all folders w/i new_data_path are trial folders
# convert jpg stacks?

# analyze videos
    skipped = 0 # number of trials skipped because data already exists
    cameras = [1,2]
    configs = [path_config_file_cam1, path_config_file_cam2]
    subs =[["c01","c1","C01","C1","Cam1","cam1","Cam01","cam01","Camera1","camera1"],["c02","c2","C02","C2","Cam2","cam2","Cam02","cam02","Camera2","camera2"]]
    trialnames = os.listdir(new_data_path)

    for trialnum,trial in enumerate(trialnames):
        trialpath = new_data_path + "/" + trial
        contents = os.listdir(trialpath)
        savepath = trialpath + "/" + "it%d"%iteration
        if os.path.exists(savepath):
            temp = os.listdir(savepath)
            if temp:
                print('There are already predicted points in iteration %d subfolders' %iteration)
                skipped += 1
                continue
        else:
            os.makedirs(savepath) # make new folder
        # get video file
        for camera in cameras:
            file = []
            for name in contents:
                if any(x in name for x in subs[camera-1]):
                    file = name
#             if len(file) > 1:
#                 exten = ['.avi','.mov','.mp4','.cine'] # if multiple files, look for video extension
#                 for name in contents:
#                     if any(x in name for x in exten):
#                         file = name
          
            if not file:
                print('Cannot locate %s video file or image folder, skipping' %trial)
                skipped += 0.5
                continue
            #analyze video
            deeplabcut.analyze_videos(configs[camera-1],[trialpath + '/' + file],destfolder = savepath,save_as_csv=True)

            # get filenames and read analyzed data
        contents = os.listdir(savepath)
        datafiles = [s for s in contents if '.h5' in s]
        cam1data = pd.read_hdf(savepath+"/"+datafiles[0])
        cam2data = pd.read_hdf(savepath+"/"+datafiles[1]) 
        dlc_to_xma(cam1data,cam2data,trial,savepath)
    print('Complete. Skipped %d trials.'%round(skipped))
            
def add_frames(path_config_file_cam1, path_config_file_cam2, data_path, iteration, frames):

    #input: config file paths, path of data to add to trainingdataset, frames-csv file where first col is trialnames and following cols are frame numbers
    # will look for 2D points file based on name (if there are multiple csv files)
    
    configs = [path_config_file_cam1[:-12], path_config_file_cam2[:-12]]
    cameras = [1,2]
    subs =[["c01","c1","C01","C1","Cam1","cam1","Cam01","cam01","Camera1","camera1"],["c02","c2","C02","C2","Cam2","cam2","Cam02","cam02","Camera2","camera2"]]
    pts = ["2Dpts","2dpts","2DPts","2dPts","pts2D","Pts2D","pts2d","points2D","Points2d","points2d","2Dpoints","2dpoints","2DPoints"]
    corr = ["correct","Correct","corrected","Corrected"]
    # read frames from csv 
    if '.csv' in frames:
        f = pd.read_csv(frames,header=None)
        trialnames = list(f.loc[:,0]) # first row of frames file must be trialnames
        picked_frames = []
        # this is disgusting code
        for row in range(f.shape[0]):
            picked_frames.append(list(f.loc[row,1:]))
        for count,row in enumerate(picked_frames):
            picked_frames[count] = [x for x in row if str(x) != 'nan'] # remove nans
        for count,row in enumerate(picked_frames):
            picked_frames[count] = [int(x) for x in row] # convert to int
    else:
        raise ValueError('frames must be a .csv file with trialnames and frame numbers')

    for camera in cameras:
        contents = os.listdir(configs[camera-1]+'/'+'labeled-data')
        if len(contents) == 1:
            dataset_name = contents[0]
            labeleddata_path = configs[camera-1] + '/' + 'labeled-data/' + dataset_name
        else:
            raise ValueError('There must be only one data set in the labeled-data folder')

        contents = os.listdir(labeleddata_path)
        h5file = [x for x in contents if '.h5' in x]
        csvfile = [x for x in contents if '.csv' in x]
        data = pd.read_hdf(labeleddata_path+'/'+h5file[0]) # read old point labels

        ## Extract selected frames from videos

        for trialnum,trial in enumerate(trialnames):
            
            frames = picked_frames[trialnum]
            frames.sort()
            frames[:] = [x - 1 for x in frames] # convert to zero index
        
        # get video file
            
            
            file = []
            relnames = []
            contents = os.listdir(data_path+"/"+trial)   
         
            for name in contents:
                if any(x in name for x in subs[camera-1]):
                    file = name
            if not file:
                raise ValueError('Cannot locate %s video file or image folder' %trial)

            # if video file is actually folder of frames
            if os.path.isdir(data_path+"/"+trial+"/"+file):
                imgpath = data_path+"/"+trial+"/"+file
                imgs = os.listdir(imgpath)
                relpath = "labeled-data/"+dataset_name+"/"
                

                for count,img in enumerate(imgs):
                    if count in frames: 
                        image = cv2.imread(imgpath+"/"+img)
                        relname = relpath + trial + "_%s.png" % str(count+1).zfill(4)
                        relnames = relnames + [relname]
                        cv2.imwrite(labeleddata_path + "/" + trial + "_%s.png" % str(count+1).zfill(4), image) # save frame 
            else:
            # file is actually a file        
            # extract frames from video and convert to png
                video = data_path+"/"+trial+"/"+file
                relpath = "labeled-data/"+dataset_name+"/"
                cap = cv2.VideoCapture(video)
                success,image = cap.read()
                count = 0
                while success:
                    if count in frames:
                        relname = relpath + trial + "_%s.png" % str(count+1).zfill(4)
                        relnames = relnames + [relname]
                        cv2.imwrite(labeleddata_path + "/" + trial + "_%s.png" % str(count+1).zfill(4), image) # save frame      
                    success,image = cap.read()
                    count += 1
                cap.release()
            
            # get 2D points file / data    
            # extract 2D points data
            datafolder = os.listdir(data_path+"/"+trial+"/it%d"%iteration)
            pointsfile = []
            # if there are multiple 2D points files, look for "corrected" in the name
            if len(datafolder) > 1:
                for r in datafolder:
                    if any(x in r for x in corr):
                        pointsfile = r
            else:
                pointsfile = datafolder[0]
    
            if not pointsfile:
                raise ValueError('Cannot locate %s 2D points file' %trial)
            
            
            
            df = pd.read_csv(data_path+'/'+trial+'/it%d'%iteration+'/'+pointsfile,sep=',',header=None)
            df = df.loc[1:,].reset_index(drop=True)
            xpos = df.iloc[frames,0+(camera-1)*2::4]
            ypos = df.iloc[frames,1+(camera-1)*2::4]
            temp_data = pd.concat([xpos,ypos],axis=1).sort_index(axis=1)
            temp_data.index = relnames
            temp_data.columns = data.columns
            data = pd.concat([data,temp_data])
            data = data.apply(pd.to_numeric) # just in case numbers are converted to strings
            
            #print('%d corrected frames extracted from '%len(frames)+'%s'%trial)
                
        data.to_hdf(labeleddata_path+'/'+h5file[0], key="df_with_missing", mode="w")
        data.to_csv(labeleddata_path+'/'+csvfile[0],na_rep='NaN')
    print('Frames from %d trials successfully added to training dataset'%len(trialnames))

