import torch.utils.data as data
import numpy as np
import os

class VideoFeatDataset(data.Dataset):
    def __init__(self, root, flist=None, frame_num=120, which_feat='both'):
        self.root = root
        self.pathlist = self.get_file_list(flist)
        self.fnum = frame_num
        self.which_feat = which_feat

    def __getitem__(self, index):
        path  = os.path.join(self.root, self.pathlist[index])
        if self.which_feat == 'vfeat':
            vfeat = self.loader(os.path.join(path, 'vfeat.npy')).astype('float32')    #visual feature
            if self.dequantize is not None:
                vfeat = self.dequantize(vfeat)
            return vfeat

        elif self.which_feat == 'afeat':
            afeat = self.loader(os.path.join(path, 'afeat.npy')).astype('float32')    #audio feature
            if self.dequantize is not None:
                afeat = self.dequantize(afeat)
            return afeat

        else:
            vfeat = self.loader(os.path.join(path, 'vfeat.npy')).astype('float32')    #visual feature
            afeat = self.loader(os.path.join(path, 'afeat.npy')).astype('float32')    #audio feature
            if self.dequantize is not None:
                vfeat = self.dequantize(vfeat)
                afeat = self.dequantize(afeat)
            return vfeat, afeat

    def __len__(self):
        return len(self.pathlist)

    def loader(self, filepath):
        return np.load(filepath)

    def get_file_list(self, flist):
        filelist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                filepath = line.strip()
                filelist.append(filepath)
        return filelist

    def dequantize(self, feat_vector, max_quantized_value=2, min_quantized_value=-2):
        """Dequantize the feature from the byte format to the float format.
        Args:
          feat_vector: the input 1-d vector.
          max_quantized_value: the maximum of the quantized value.
          min_quantized_value: the minimum of the quantized value.
        Returns:
          A float vector which has the same shape as feat_vector.
        """
        assert max_quantized_value > min_quantized_value
        quantized_range = max_quantized_value - min_quantized_value
        scalar = quantized_range / 255.0
        bias = (quantized_range / 512.0) + min_quantized_value
        return feat_vector * scalar + bias
