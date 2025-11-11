import os, torch
import numpy as np
from torch.utils.data import Dataset

class ViMoDataset(Dataset): # dont use num_workers in dataloader, its not working!
    N_VIEWS = 9 # class_vars, total view must be 9 accoding to dataset
    VAL_IDX_MAP = {} # hash map['file'->list('val_idx')] to remember val_idxs
    # acces via class_name to modify & via self to read as safety measure
    """
    Parameters:
        - file_list: list of files from which to load data from
        - train: if enabled, each view of 2d pose is considered as single training data paired with resp 3d motion
        - conf_thresh: set keypoints with low confidence threshold to zero i.e. mark as missing
        - t_views: no.of views to consider for training per file
        - v_views: no.of views to consider for validation per file
    Functions:
        - pose_preprocessor(): fn to stabilze (i.e. normalize) 2d poses that represent image pixel coordinated sys
    """
    def __init__(self, file_list, t_views=8, conf_thresh=0.2, train=True):
        self.file_list = file_list
        self.train = train
        self.conf_thresh = conf_thresh
        if (t_views > self.N_VIEWS and isinstance(t_views, int)):
            raise RuntimeError("Max views per sample is 9! (enter proper integer)")
        if (t_views < self.N_VIEWS and self.VAL_IDX_MAP=={}):
            raise RuntimeError("Val_Idx_Map need to be pre-initialized to make use multiple workers without race condition")
        self.t_views = t_views # no.of views per p2d in training
        self.v_views = self.N_VIEWS - t_views # no.of views per p2d in training
        
    def pose_preprocessor(self, p2d):
        p2d[np.isnan(p2d)] = 0 # reset nan values
        
        # Centering: Subtract the coordinates of a reference joint (pelvis)
        root = p2d[:,:1] # pelvis (joint id=0)
        body = p2d[:,1:] # rest all joints
        body[:,:,:2] = body[:,:,:2] - root[:,:,:2] # (S,16,2) - (S,1,2)

        # Confidence Filtering: Set keypoints with low confidence to zero
        body[body[:,:,2] < self.conf_thresh] = 0
        return np.concatenate((root, body), axis=1)
        
    def __len__(self):
        if self.train:
            return len(self.file_list) * self.t_views
        else:
            return len(self.file_list) * self.v_views

    def __getitem__(self, idx):
        file_idx = idx
        if self.train:
            file_idx = idx // self.t_views
        else:
            file_idx = idx // self.v_views
        
        file_path = self.file_list[file_idx]
        file_name = os.path.basename(file_path)
        data = np.load(file_path)

        view_idx = 0
        if self.train: # training dataset loader    
            view_idx = idx % self.t_views
            if self.t_views < self.N_VIEWS:
                all_idx = list(np.arange(self.N_VIEWS))
                #print("val_idx for this file:", VAL_IDX_MAP[file_name])
                idx_map = [idx for idx in all_idx if idx not in self.VAL_IDX_MAP[file_name]]
                #print(f"remap order: {list(enumerate(idx_map))}")
                view_idx = idx_map[view_idx]

        else:  # validation dataset loader
            view_idx = idx % self.v_views
            view_idx = self.VAL_IDX_MAP[file_name][view_idx]
            
        #print(f"file_name={file_name},  p2d.shape={data['p2d_cond'].shape}, fetching view_idx={view_idx}")
        p2d = data['p2d_cond'][view_idx]  # (S, 17, 3) expected
        m3d = data['m3d_gt']    # (S, 151) expected
        self.pose_preprocessor(p2d)
        if np.isnan(m3d).any():
            raise ValueError(f"3D motions from preprocessed data(file={file_name}) contained nan values!")

        # to tensors: (S,17,3) -> float32 ; (S,151) -> float32
        p2d_t = torch.from_numpy(p2d).float()
        m3d_t = torch.from_numpy(m3d).float()
        return {'m3d': m3d_t, 'p2d': p2d_t, 'view_idx':view_idx, 'file_name': file_name}
    
    @classmethod
    def get_train_val_pair(cls, file_list, t_views=8, conf_thresh=0.2):
        for file_path in file_list: # to avoid resetting multiple times in train_ds and val_ds
            file_name = os.path.basename(file_path)
            ViMoDataset.VAL_IDX_MAP[file_name] = random.sample(range(cls.N_VIEWS), cls.N_VIEWS - t_views)
        return (cls(file_list, t_views, conf_thresh, train=True), cls(file_list, t_views, conf_thresh, train=False))
