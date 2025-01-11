import glob
import os.path as osp
import re
import numpy as np

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class etri_test1(ImageDataset):

    dataset_dir = "etri_test1"
    dataset_name = "etri_test1"

    def __init__(self, root='dataset', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train_st')
        
        # self.query_dir = osp.join(self.dataset_dir, 'image_query')
        # self.gallery_dir = osp.join(self.dataset_dir, 'image_gallery')
        
        self.target_id = int(input('\n\ntarget id: '))
        self.frame_range = 150000
        self.query_dir = osp.join(self.dataset_dir, 'c03')
        self.gallery_dir = osp.join(self.dataset_dir, 'c01')

        self.query_frame = 0

        # f_path = '202311261500_202311261520'
        # a = 69      # query pid
        # b = 'cq03'   # query camid
        # c = 142828  # query frame
        # d = 'c01'   # target camid
        
        # self.query_dir = osp.join(self.dataset_dir, f'{f_path}/{a:05d}_{b}_{c:07d}_t_{d}/query')
        # self.gallery_dir = osp.join(self.dataset_dir, f'{f_path}/{a:05d}_{b}_{c:07d}_t_{d}/gallery')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False, is_query=True)
        gallery = self.process_dir(self.gallery_dir, is_train=False, is_test=True)

        super(etri_test1, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, is_query=False, is_test=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c([\d]+)_([\d]+)')

        data = []
        frames = []
        for img_path in img_paths:
            pid, camid, frame = map(int, pattern.search(img_path).groups())

            if pid == -1: continue  # junk images are just ignored
#             assert 0 <= pid <= 776
#             assert 1 <= camid <= 20
#             assert 0 <= frame
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                frame = self.dataset_name + "_" + str(frame) 
                
            if is_query:
                if pid == self.target_id:
                    frames.append(frame)
                else:
                    continue
            
            if is_test:
                if frame < self.query_frame - self.frame_range or frame > self.query_frame + self.frame_range:
                    continue
                
            data.append((img_path, pid, camid, frame))

        if is_query:
            self.query_frame = np.mean(frames)

        return data