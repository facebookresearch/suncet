import torch.utils.data as data
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor

from PIL import Image,  ImageStat
import numpy as np

import os
import os.path
import sys
from random import randint

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

"""
def crop_loader(path, coords):
    import accimage
    return accimage.Image(path).crop(coords)
"""


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetDataframe(data.Dataset):
    """A dataset using a dataframe  ::

        img | cords | class

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root_dir, df, transform=None, loader=default_loader, target_transform=None):
        self.cords_frame = df
        self.root_dir = root_dir
        self.samples = list(zip(df['img_path'],df['target'],df[['x1','y1','x2','y2']].values.tolist()))
        self.target = df['target'].values.tolist()
        if len(df) == 0:
            raise(RuntimeError("Found 0 files in dataframe"))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # self.classes =

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, coords = self.samples[index] # coords [x1,y1,x2,y2]
        #print(index, path, target)
        sample = self.loader(os.path.join(self.root_dir, path))

        args = (sample, coords)

        if self.transform is not None:
            sample = self.transform(args)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
                     
        try:
            return len(self.cords_frame)
        except:
            return len(self.samples)
            


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



class Crop(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __call__(self, params):
        sample, coords = params
        sample = sample.crop(coords)#[coords[1]: coords[3],
                      #coords[0]: coords[2]]
        return sample


def cut_down(img, width, height):
    """ find black frame at the bottom
    Return y2
    """
    zeros = np.asarray(img)[:,
            (int(width/4), int(1 * width/3), int(width/2),(int(2*width/3))),
                :].sum(axis=1).sum(axis=1)

    row0 = np.nonzero(zeros == 0)
    if len(row0[0]) > 0:
        minumun = np.min(row0) - 1
        return width,  minumun
    else:
        return width, height


class SquareCrop(object):
    """Rescale the image in a sample to a given size, fitted in a square
    """
    def __call__(self, params):
        sample, coords = params
        width, height = sample.size
        width, height = cut_down(sample, width, height)

        x1, y1, x2, y2 = coords
        x = x2 - x1
        y = y2 - y1

        padding = (0, 0)
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

        if x > y :
            y2_ = int((y1 + y2 + x)/2.)
            y1_ = int((y1 + y2 - x)/2.)
            x1_ = x1
            x2_ = x2

        elif x <= y:
            x2_ = int((x1 + x2 + y)/2.)
            x1_ = int((x1 + x2 - y)/2.)
            y1_ = y1
            y2_ = y2

        if y1_ < 0 :
            pad_top = - y1_
            y1_ = 0
        if y2_ > height:
            pad_bottom = y2_ - height
            y2_ = height
        if x1_ < 0 :
            pad_left = - x1_
            x1_ = 0
        if x2_ > width:
            pad_right = x2_ - width
            x2_ = width

        padding = (int((pad_left + pad_right)/2.),  int((pad_bottom + pad_top)/2.))
        coords = (x1_, y1_, x2_, y2_)

        """
        print(padding)
        print(coords)
        print((x2_ - x1_, y2_ - y1_))
        """

        sample = sample.crop(coords)

        if sum(padding) > 0 :
            # tuple(ImageStat.Stat(sample).mean)
            pad = transforms.Pad(padding, fill=(100,100,100), padding_mode='constant') # padding_mode IN ['constant', 'edge', 'reflect', 'symmetric']
            sample = pad(sample)

        #sample.save("/model/test_square/img/" + str(randint(0,1e6)) + ".jpg")
        return sample

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, coords = self.samples[index] # coords [x1,y1,x2,y2]
        sample = self.loader(path,coords)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
