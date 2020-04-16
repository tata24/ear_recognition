from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, root, transforms=None, target_transform=None):
        super(MyDataset, self).__init__()
        self.img_list = []
        self.labels = []
        cls1_dir = root + '/Aran'
        cls2_dir = root + '/Axian'
        cls3_dir = root + '/Ayi'
        self.get_file_list(cls1_dir, 0)
        self.get_file_list(cls2_dir, 1)
        self.get_file_list(cls3_dir, 2)
        self.transforms = transforms
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        if self.transforms is not None:
            img = self.transforms(img)  # 是否进行transforms
        return img, self.labels[index]

    def __len__(self):
        return len(self.img_list)

    def get_file_list(self, dir, label):
        if os.path.isfile(dir) and os.path.splitext(dir)[1] == '.jpg':
            self.img_list.append(dir)
            self.labels.append(label)
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                new_dir = os.path.join(dir, s)
                self.get_file_list(new_dir, label)






