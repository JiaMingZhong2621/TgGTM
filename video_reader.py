import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random
from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, CenterCrop

"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []
    def add_vid(self, paths, gt_a,prompt=None):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)
    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)

"""Dataset for few-shot videos, which returns few-shot tasks. """
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.get_item_counter = 0

        self.data_dir = args.path
        self.seq_len = args.seq_len
        self.train = True
        self.tensor_transform = transforms.ToTensor()
        self.img_size = 224

        self.annotation_path = args.traintestlist

        self.way = args.way
        self.shot = args.shot
        self.query_per_class = args.query_per_class
        self.query_per_class_test = args.query_per_class_test
        
        self.train_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self._select_fold()
        self.read_dir()

    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
    def setup_transforms(self):
        video_transform_list = []
        video_test_list = []
            
        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        video_transform_list.append(RandomHorizontalFlip())
        video_transform_list.append(RandomCrop(self.img_size))

        video_test_list.append(CenterCrop(self.img_size))

        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)
    
    """Loads all videos into RAM from an uncompressed zip. Necessary as the filesystem has a large block size, which is unsuitable for lots of images. """
    """Contains some legacy code for loading images directly, but this has not been used/tested for a while so might not work with the current codebase. """
    def read_dir(self):
        #loading train data
        self.data_dir_train = os.path.join(self.data_dir,"train")
        class_folders = os.listdir(self.data_dir_train)
        if self.args.dataset == 'ssv2':
            class_folders = sorted(class_folders, key=lambda x: int(x))
        else:
            class_folders.sort()
        self.class_folders = class_folders
        for class_folder in class_folders:
            video_folders = os.listdir(os.path.join(self.data_dir_train, class_folder))
            video_folders.sort()
            for video_folder in video_folders:
                c = self.get_train_or_test_db(video_folder)
                if c == None:
                    continue
                imgs = os.listdir(os.path.join(self.data_dir_train, class_folder, video_folder))

                if len(imgs) < self.seq_len:
                    continue            
                imgs.sort()
                paths = [os.path.join(self.data_dir_train, class_folder, video_folder, img) for img in imgs]
                paths.sort()
                class_id =  class_folders.index(class_folder)
                c.add_vid(paths, class_id)
        print("loaded {}".format(self.data_dir_train))
    
        #loading test data
        self.data_dir_test = os.path.join(self.data_dir,"test")
        class_folders = os.listdir(self.data_dir_test)
        if self.args.dataset == 'ssv2':
            class_folders = sorted(class_folders, key=lambda x: int(x))
        else:
            class_folders.sort()
        self.class_folders = class_folders
        for class_folder in class_folders:
            video_folders = os.listdir(os.path.join(self.data_dir_test, class_folder))
            video_folders.sort()
            for video_folder in video_folders:
                c = self.get_train_or_test_db(video_folder)
                if c == None:
                    continue
                imgs = os.listdir(os.path.join(self.data_dir_test, class_folder, video_folder))
                if len(imgs) < self.seq_len:
                    continue            
                imgs.sort()
                paths = [os.path.join(self.data_dir_test, class_folder, video_folder, img) for img in imgs]
                paths.sort()
                class_id =  class_folders.index(class_folder)
                c.add_vid(paths, class_id)
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))
        
    """ return the current split being used """
    def get_train_or_test_db(self, split=None):
        if split is None:
            get_train_split = self.train
        else:
            if split in self.train_test_lists["train"]:
                get_train_split = True
            elif split in self.train_test_lists["test"]:
                get_train_split = False
            else:
                return None
        if get_train_split:
            return self.train_split
        else:
            return self.test_split
    
    """ load the paths of all videos in the train and test splits. """ 
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name, 3)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()

                if "kinetics" == self.args.dataset:
                    data = [x.replace('\n', 'frames') for x in data]
                elif self.args.dataset !='ssv2' and self.args.dataset !='ucf_crime':
                    data = [x.replace('\n', '_') + 'frames' for x in data]
                data = [x.replace(' ', '_') for x in data]
                data = [x.strip().split(" ")[0] for x in data]
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data] 

                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists
    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        return len(c)
   
    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes
    
    """Loads a single image from a specified path """
    def read_single_image(self, path):
        with Image.open(path) as i:
            i.load()
            return i
    
    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx) 
        n_frames = len(paths)
        if n_frames == self.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            if self.train:
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames-1 -excess_pad, n_frames-1)
            else:
                start = 1
                end = n_frames - 2
    
            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0
            else:
                pass
    
            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
            
            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]
        imgs = [self.read_single_image(paths[i]) for i in idxs]
            
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)

        return imgs


    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        #select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.query_per_class
        else:
            n_queries = self.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        support_real_class = []
        real_target_labels= []
        for bl, bc in enumerate(batch_classes):
            
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.shot + n_queries)

            for idx in idxs[0:self.shot]:
                vid = self.get_seq(bc, idx)
                support_set.append(vid)
                support_labels.append(bl)
                support_real_class.append(bc)
            for idx in idxs[self.shot:]:
                vid= self.get_seq(bc, idx)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)

        s = list(zip(support_set, support_labels,support_real_class))
        random.shuffle(s)
        support_set, support_labels,support_real_class = zip(*s)
        
        t = list(zip(target_set, target_labels,real_target_labels))
        random.shuffle(t)
        target_set, target_labels,real_target_labels = zip(*t)
        
        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        support_real_class = torch.FloatTensor(support_real_class)
        batch_classes = torch.FloatTensor(batch_classes) 
        real_target_labels = torch.FloatTensor(real_target_labels)       
        
        return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels, 
                "support_real_class":support_real_class, "real_target_labels": real_target_labels}


