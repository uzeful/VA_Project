import numpy as np
import os
import shutil
import pdb

root = '/data/datasets/yt8m'
sample_space = 40
frame_num = 120
count = 0

train_file = 'train.txt'
test_file = 'test.txt'

train_fid = open(train_file, 'w')
test_fid = open(test_file, 'w')

pathlist = os.listdir(root)
for path in pathlist:
    fullpath  = os.path.join(root, path)
    vfeat = np.load(os.path.join(fullpath, 'vfeat.npy'))
    afeat = np.load(os.path.join(fullpath, 'afeat.npy'))
    if vfeat.shape[0]!=frame_num or afeat.shape[0]!=frame_num:
        #pdb.set_trace()
        shutil.rmtree(fullpath)
        print('not valid path:{0}'.format(path))
    else:
        count = count + 1
        if count % sample_space == 0:
            test_fid.write('{}\n'.format(path))
        else:
            train_fid.write('{}\n'.format(path))
train_fid.close()
test_fid.close()
