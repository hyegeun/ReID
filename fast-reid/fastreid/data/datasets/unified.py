import glob
import os.path as osp
import re
import random

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class unified(ImageDataset):

    dataset_dir = "aa"
    dataset_name = "aa"

    def __init__(self, dataset_path, query_id, initial_cam, neighbor_cam, threshold, time_zone_dict, fusion_dict, root='datasets', **kwargs):
        print(dataset_path, query_id, neighbor_cam)
        self.dataset_dir = osp.join(root, dataset_path)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, f'query_dir/{threshold}/{query_id:04d}_c{initial_cam:03d}')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, is_train=True)
        query, max_frame, max_cam = self.process_dir_query(self.query_dir)
        gallery = self.process_dir(self.gallery_dir, max_frame, max_cam, fusion_dict, neighbor_cam = neighbor_cam)

        super(unified, self).__init__(train, query, gallery, **kwargs)


    def process_dir(self, dir_path, max_frame=None, max_cam=None, fusion_dict=None, neighbor_cam=None, is_train=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)_([\d]+)')

        gallery_time_dict = {}

        data = []

        for img_path in img_paths:
            pid, camid, frame = map(int, pattern.search(img_path).groups())
            
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 776
            # assert 1 <= camid <= 20
            assert 0 <= frame
            camid -= 1  # index starts from 0

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                frame = self.dataset_name + "_" + str(frame)

            else: # test인 경우 
                cam1 = f'c{max_cam:03d}'
                if camid+1 not in neighbor_cam: # neighbor cam에 있는 데이터들만 가져옴 
                    continue
                if frame <= max_frame: # 일단 query보다 일찍 나오는 데이터들은 제거 
                    continue

                # if fusion_dict != None: # FusionNet을 사용하는 경우 
                #     cam2 = f'c{camid+1:03d}'
                #     cam_pair = f'{cam1}-{cam2}'
                #     bin_idx = abs(frame - max_frame) // 100
                #     if bin_idx < 0 or bin_idx >= len(fusion_dict[cam_pair]): # bin에 안 나오면 fusion_dict에서 에러뜨므로 제거 
                #         continue

                #     if fusion_dict[cam_pair][bin_idx] < 1e-10: # 값보다 낮은 확률을 가지면 제거 
                #         # print(cam_pair, bin_idx)
                #         continue
                # else:
                if frame > max_frame + 4000:
                    continue

            if pid in gallery_time_dict.keys():
                if camid in gallery_time_dict[pid].keys():
                    gallery_time_dict[pid][camid].append((img_path, pid, camid, frame, 0))
                else:
                    gallery_time_dict[pid][camid] = [(img_path, pid, camid, frame, 0)]
            else:
                gallery_time_dict[pid] = {}
                gallery_time_dict[pid][camid] = [(img_path, pid, camid, frame, 0)]
            
            data.append((img_path, pid, camid, frame, 0))

        # for pid in gallery_time_dict.keys():
        #     for camid in gallery_time_dict[pid].keys():
        #         random_element = random.choice(gallery_time_dict[pid][camid])
        #         data.append(random_element)


        return data


    def process_dir_query(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)_([\d]+)')

        data = []
        max_frame = 0

        for img_path in img_paths:
            pid, camid, frame = map(int, pattern.search(img_path).groups())
            
            # exit_frame = time_zone_dict[pid][camid]

            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 776
            # assert 1 <= camid <= 20
            assert 0 <= frame
            camid -= 1  # index starts from 0

            if frame > max_frame:
                max_frame = frame
                max_cam = camid+1
                
            data.append((img_path, pid, camid, frame, 0))

        # random_data = random.choice(data)
        # return [random_data], max_frame, max_cam

        return data, max_frame, max_cam