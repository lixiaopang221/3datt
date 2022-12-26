import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
from glob import glob

class Load():
    def __init__(self, data_root, samples_txt, name = 'load'):
        self.name = name
        self.path = data_root
        with open(samples_txt) as file:
            self.samples = file.read().splitlines()
            Load.sample_names = self.samples

    def load(self, samples_batch = []):
        if 'brats2015' in self.samples[0].lower():
            return self.load_mha(samples_batch)
        else:
            return self.load_nii(samples_batch)
            
    def load_nii(self, samples_batch = []):
        'data_format: NDHWC'
        samples = self.samples
        if len(samples_batch):
            samples = samples_batch
            self.samples = samples
        self.imgs = []
        self.labels = []
        self.boxes = []
        for sample in samples:
            img_names = os.listdir(os.path.join(self.path, sample))
            if len(img_names) < 4:
                continue
            print('load {0}'.format(sample))
            for img_modal in img_names:
                path = os.path.join(self.path, sample, img_modal)
                if 't1.nii' in img_modal:
                    t1 = nib.load(path).get_data()
                    self.boxes.append(self.get_bounding_box(np.transpose(t1, [2,1,0])))
                    t1 = self.img_normalize(t1)
                elif 't1ce.nii' in img_modal:
                    t1ce = nib.load(path).get_data() 
                    t1ce = self.img_normalize(t1ce)
                elif 't2.nii' in img_modal:
                    t2 = nib.load(path).get_data()
                    t2 = self.img_normalize(t2)
                elif 'flair.nii' in img_modal:
                    flair = nib.load(path).get_data() 
                    flair = self.img_normalize(flair)
                elif 'seg.nii' in img_modal:
                    seg = nib.load(path).get_data() 
                    # whole tumor: 2;  tumor core: 1; enhancing tumor core: 4 
                    label_or = self.label_convert(seg, (1,2,4), (2,1,3))
                    label_wt = self.label_convert(seg, (1,2,4), (1,1,1))
                    label_tc = self.label_convert(seg, (1,2,4), (1,0,1))
                    label_et = self.label_convert(seg, (1,2,4), (0,0,1))
                    label = np.stack([label_or,label_wt,label_tc,label_et], axis = 3)
                    self.labels.append(label)
            img = np.stack([t1,t1ce,t2,flair], axis = 3)
            self.imgs.append(img)
        self.imgs = np.stack(self.imgs)
        self.imgs = np.transpose(self.imgs, [0,3,2,1,4])
        if len(self.labels):
            self.labels = np.stack(self.labels)
            self.labels = np.transpose(self.labels, [0,3,2,1,4])
        return self.imgs, self.labels
    
    def load_mha(self, samples_batch = []):
        'data_format: NDHWC'
        samples = self.samples
        if len(samples_batch):
            samples = samples_batch
            self.samples = samples
        self.imgs = []
        self.labels = []
        self.boxes = []
        for sample in samples:
            filenames = glob(f'{self.path}/{sample}/*/*.mha')
            print('load {0}'.format(sample))
            for filename in filenames:
                if 'T1.' in filename:
                    t1 = sitk.GetArrayFromImage(sitk.ReadImage(filename))
                    self.boxes.append(self.get_bounding_box(t1))
                    t1 = self.img_normalize(t1)
                elif 'T1c.' in filename:
                    t1ce = sitk.GetArrayFromImage(sitk.ReadImage(filename))
                    t1ce = self.img_normalize(t1ce)
                elif 'T2.' in filename:
                    t2 = sitk.GetArrayFromImage(sitk.ReadImage(filename))
                    t2 = self.img_normalize(t2)
                elif 'Flair.' in filename:
                    flair = sitk.GetArrayFromImage(sitk.ReadImage(filename))
                    flair = self.img_normalize(flair)
                elif 'OT.' in filename:
                    seg = sitk.GetArrayFromImage(sitk.ReadImage(filename))
                    # whole tumor: 2;  tumor core: 1; enhancing tumor core: 4 
                    label_or = self.label_convert(seg, (1,2,3,4), (2,1,2,3))
                    label_wt = self.label_convert(seg, (1,2,3,4), (1,1,1,1))
                    label_tc = self.label_convert(seg, (1,2,3,4), (1,0,1,1))
                    label_et = self.label_convert(seg, (1,2,3,4), (0,0,0,1))
                    label = np.stack([label_or,label_wt,label_tc,label_et], axis = 3)
                    self.labels.append(label)
            img = np.stack([t1,t1ce,t2,flair], axis = 3)
            self.imgs.append(img)
        self.imgs = np.asarray(self.imgs)
        self.labels = np.asarray(self.labels)
        return self.imgs, self.labels
                

    def label_convert(self, inputs, label_list, label_target):
        img = np.zeros_like(inputs)
        for L, T in zip(label_list, label_target):
            img[inputs == L] = T
        return img


    def img_normalize(self, inputs):
        if 'nor' in self.path:
            return inputs    #--- no normalize
        pixels = inputs[inputs > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (inputs - mean) / std
        out_random = np.random.normal(0, 1, size = inputs.shape)
        out[inputs == 0] = out_random[inputs == 0]
        # out[inputs == 0] = 0
        return out.astype('float32')

    def get_patch(self, img_list, patch_shape):
        '''
        img_list = [imgs, labels, weights]
        imgs, labels: NDHWC.
        return the random patch of the imgs and labels.
        '''
        
        N, D, H, W, C = img_list[0].shape
        d_max = D - patch_shape[0]
        h_max = H - patch_shape[1]
        w_max = W - patch_shape[2]
        point_d = np.random.randint(d_max)
        point_h = np.random.randint(20, h_max - 20 + 1)
        point_w = np.random.randint(20, w_max - 20 + 1)
        out_list = []
        for img in img_list:
            img_patch = img[:,point_d:point_d + patch_shape[0],
                              point_h:point_h + patch_shape[1],
                              point_w:point_w + patch_shape[2],...]
            out_list.append(img_patch)
        return out_list

        
    def load_patch_from_sys(self, batch_size):
        'patch 已经保存'
        modal_filenames = []
        label_filenames = []
        for sample in self.samples:
            names = os.listdir(f'{self.path}/{sample}/modal')
            for name in names:
                modal_filenames.append(f'{self.path}/{sample}/modal/{name}')
                label_filenames.append(f'{self.path}/{sample}/label/{name}')
        idx = list(range(len(modal_filenames)))
        modal_filenames = np.asarray(modal_filenames)
        label_filenames = np.asarray(label_filenames)
        np.random.shuffle(idx)
        for i in range(0, len(modal_filenames), batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_modal_filenames = modal_filenames[batch_idx]
            batch_label_filenames = label_filenames[batch_idx]
            batch_modals = self.load_one_batch_of_patch(batch_modal_filenames)
            batch_modals = self.img_fill_zero(batch_modals)
            batch_labels = self.load_one_batch_of_patch(batch_label_filenames)
            yield batch_modals, batch_labels

    def load_one_batch_of_patch(self, filenames):
        imgs = []
        for filename in filenames:
            img = nib.load(filename).get_data()
            imgs.append(img)
        return np.asarray(imgs)
        
    def img_fill_zero(self, img):
        fill_random = np.random.normal(0, 1, size = img.shape)
        img[img==0] = fill_random[img==0]
        return img
        
    def get_batch_of_patch_in_bounding_box(self, batch_size, p_shape):
        n = len(self.imgs)
        for idx in range(0, n, batch_size):
            imgs, labels = [], []
            i = 0
            while(i < batch_size):
                idxi = idx + i
                if idxi < n:
                    topleft = []
                    idxmin, idxmax = self.boxes[idxi]
                    idxmax = idxmax - p_shape
                    for j in range(len(p_shape)):
                        topleft.append(np.random.randint(idxmin[j], idxmax[j]+1))
                    topleft = np.asarray(topleft, np.int)
                    d0_slice = slice(topleft[0], topleft[0] + p_shape[0])
                    d1_slice = slice(topleft[1], topleft[1] + p_shape[1])
                    d2_slice = slice(topleft[2], topleft[2] + p_shape[2])
                    img_patch = self.imgs[idxi,d0_slice, d1_slice, d2_slice,...]
                    label_patch = self.labels[idxi,d0_slice, d1_slice, d2_slice,...]
                    if label_patch[...,2].sum() < 10:
                        continue
                    imgs.append(img_patch)
                    labels.append(label_patch)
                    i += 1
                else:
                    break
            yield np.array(imgs), np.array(labels)
        
    def get_batch_of_patch(self, batch_size, p_shape):
        n = len(self.imgs)
        for idx in range(0, n, batch_size):
            imgs, labels = [], []
            i = 0
            while(i < batch_size):
                idxi = idx + i
                if idxi < n:
                    topleft = []
                    idxmin, idxmax = [0,20,20], [150,220,220]
                    idxmax = idxmax - np.array(p_shape)
                    for j in range(len(p_shape)):
                        topleft.append(np.random.randint(idxmin[j], idxmax[j]+1))
                    topleft = np.asarray(topleft, np.int)
                    d0_slice = slice(topleft[0], topleft[0] + p_shape[0])
                    d1_slice = slice(topleft[1], topleft[1] + p_shape[1])
                    d2_slice = slice(topleft[2], topleft[2] + p_shape[2])
                    img_patch = self.imgs[idxi,d0_slice, d1_slice, d2_slice,...]
                    label_patch = self.labels[idxi,d0_slice, d1_slice, d2_slice,...]
                    if label_patch[...,2].sum() < 10:
                        continue
                    imgs.append(img_patch)
                    labels.append(label_patch)
                    i += 1
                else:
                    break
            yield np.array(imgs), np.array(labels)
                
    def get_bounding_box(self, img, p_shape = [96,96,96]):
        'return the topleft point and downright ponit of the bounding box'
        idxs = np.nonzero(img)
        idxmin = np.min(idxs, axis = 1)
        idxmax = np.max(idxs, axis = 1)
        for i in range(len(p_shape)):
            l_shape = p_shape[i]
            l_minmax = idxmax[i] - idxmin[i]
            if l_minmax < l_shape:
                idxmin[i] = idxmin[i] - np.ceil((l_shape - l_minmax)/2)
                idxmax[i] = idxmax[i] + np.ceil((l_shape - l_minmax)/2)
        return idxmin, idxmax


if __name__ == '__main__':
    loader = Load(r"E:\yan\dataset\BraTS\BRATS2015", r'../../data\brats15\XY18.txt')
    imgs, labels = loader.load()
    print(np.shape(imgs))
    print(np.shape(labels))
    print(Load.sample_names)