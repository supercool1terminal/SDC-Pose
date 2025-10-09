import glob #Finds a wildcard for the file path
import math #Provides mathematical operation functions
import os  #Operating system related path operations

import _pickle as cPickle  #Serialize and deserialize Python objects
import pickle
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms  #Image standardization, conversion
from PIL import Image  #Image conversion
from torch.utils.data import Dataset

from utils.data_utils import fill_missing, get_bbox, load_composed_depth, load_depth, rgb_add_noise

class TrainingDataset(Dataset):
    def __init__(self, image_size, sample_num, data_dir, data_type='real', num_img_per_epoch=-1, threshold=0.2):
        self.data_dir = data_dir
        self.data_type = data_type
        self.threshold = threshold
        self.num_img_per_epoch = num_img_per_epoch
        self.img_size = image_size
        self.sample_num = sample_num    #Initializes the data set, specifying the size of the image, the number of samples per training, the data type ('real' or 'syn'), and other relevant parameters

        if data_type == 'syn':  #Select the type of training data if it is synthetic data ('syn') or real data ('real_withLabel')
            img_path = 'camera/train_list.txt'
            model_path = 'obj_models/camera_train.pkl'
            self.intrinsics = [577.5, 577.5, 319.5, 239.5]      #Store the camera's internal parameters
        elif data_type == 'real_withLabel':
            img_path = 'real/train_list.txt'
            model_path = 'obj_models/real_train.pkl'
            self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        else:
            assert False, 'wrong data type of {} in data loader !'.format(data_type)

        self.img_list = [os.path.join(img_path.split('/')[0], line.rstrip('\n'))
                        for line in open(os.path.join(self.data_dir, img_path))]
        self.img_index = np.arange(len(self.img_list))

        self.models = {}        #Load the saved 3D model data, stored in the Pickle file
        with open(os.path.join(self.data_dir, model_path).replace("\\","/"), 'rb') as f:
            self.models.update(cPickle.load(f))

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.colorjitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.3) #Color dithering method for image enhancement
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])  #Standard image conversion (ToTensor and standardization)
        print('{} images found.'.format(len(self.img_list)))
        print('{} models loaded.'.format(len(self.models)))

    def __len__(self): #Returns the data set size. If num_img_per_epoch for each training is set to -1, return the number of all images, otherwise return the set number of samples
        if self.num_img_per_epoch == -1:
            return len(self.img_list)
        else:
            return self.num_img_per_epoch

    def reset(self): #Reinitialize the sample order before each epoch to ensure sample randomization
        assert self.num_img_per_epoch != -1
        num_img = len(self.img_list)
        if num_img <= self.num_img_per_epoch:
            self.img_index = np.random.choice(num_img, self.num_img_per_epoch)
        else:
            self.img_index = np.random.choice(num_img, self.num_img_per_epoch, replace=False)  #If the number of samples is greater than the set number of samples, the samples that do not duplicate are randomly selected

    def __getitem__(self, index):  #Load a sample RGB image, depth map, label, etc., for pre-processing and data enhancement

        # print(f"Worker PID: {os.getpid()}, Loading sample {index}, img_path: {self.img_list[self.img_index[index]]}")

        img_path = os.path.join(self.data_dir, self.img_list[self.img_index[index]])
        if self.data_type == 'syn':     #Select the appropriate depth map loading method based on data_type (synthetic data or real data)
            depth = load_composed_depth(img_path)
        else:
            depth = load_depth(img_path)
        if depth is None:  #If the loaded depth chart is empty, a new sample is randomly selected for loading. This section ensures that the depth map for each sample is valid.
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        #Fill in missing values in the depth map, self.norm_scale is the standardized scale, 1 indicates the filling strategy
        depth = fill_missing(depth, self.norm_scale, 1)

        #(mask) Load labels for the corresponding image (including object instance, class ID, bounding box, etc.)
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = pickle.load(f)   #Load the tag data corresponding to the image through pickle (including instance ID, category ID, bounding box, etc.)
        num_instance = len(gts['instance_ids'])  #The number of instances in the tag (i.e. the number of objects in the current image)
        assert(len(gts['class_ids'])==len(gts['instance_ids']))
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2] #480*640  Load a depth map corresponding to the image (to identify the location of each object)

        idx = np.random.randint(0, num_instance)  #Randomly select an instance idx from the object instances
        cat_id = gts['class_ids'][idx] - 1 # convert to 0-indexed  The class ID of the instance, converted to 0-indexed by subtracting 1
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx]) #Calculate the rectangular area of the object based on the bounding box of the instance (rmin, rmax, cmin, cmax)
        mask = np.equal(mask, gts['instance_ids'][idx])
        mask = np.logical_and(mask , depth > 0)  #A mask is generated based on the instance ID and depth map. The mask belongs to the location of the instance, and the depth value is greater than 0

        # choose sample point
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]  # nonzero index Gets an index of all non-zero pixels within a specified rectangular area
        if len(choose) <= 0: #If no pixels are selected, a new sample is selected

            print(f"warning: no valid point for index {index},img_path: {img_path}")

            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        if len(choose) <= self.sample_num: #If the number of pixels in this instance is less than or equal to the predetermined number of sampling points (sample_num), the sample_num pixels are randomly selected
            choose_idx = np.random.choice(len(choose), self.sample_num)
        else:                               #If the number of pixels exceeds sample_num, the sample_num pixels are randomly selected without replacement
            choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
        choose = choose[choose_idx]

        # pts 3D coordinate point
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics  #Calculate the 3D point cloud coordinates using the depth map and camera parameters (cam_fx, cam_fy, cam_cx, cam_cy)
        pts2 = depth.copy() / self.norm_scale
        pts0 = (self.xmap - cam_cx) * pts2 / cam_fx
        pts1 = (self.ymap - cam_cy) * pts2 / cam_fy        #pts0, pts1, pts2 represent the transformation of the image coordinates on the X, Y, and Z axes, respectively. They are then merged into a 3D point cloud
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3
        pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :] #Keep only the 3D point cloud of the cropped area and transform it into a flat one-dimensional array, and finally select the points of interest according to the choose index

        # (add noise) A small amount of Gaussian noise is added to each 3D point, which contributes to the robustness of the model
        pts = pts + np.clip(0.001*np.random.randn(pts.shape[0], 3), -0.005, 0.005)

        # rgb
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        # crop
        rgb = rgb[:, :, ::-1] #480*640*3
        rgb = rgb[rmin:rmax, cmin:cmax, :]  #Load the RGB image and crop it to the previously calculated rectangular area
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)  #Use cv2.resize to resize the image for subsequent training and processing
        # data augmentation
        rgb = self.colorjitter(Image.fromarray(np.uint8(rgb))) #ColorJitter is used for image color dithering to increase the generalization ability of the model
        rgb = np.array(rgb)
        rgb = rgb_add_noise(rgb)            #Add noise (rgb_add_noise) for data enhancement
        rgb = self.transform(rgb)           #Finally, transform was used to standardize the image, making it conform to the input requirements of the neural network
        # update choose
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64) # Map the selected 2D image coordinates to the cropped and scaled image

        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(pts) # N*3
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['choose'] = torch.IntTensor(choose).long()
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()  #Encapsulate 3D point clouds, RGB images, selected pixel indexes, and category labels into a dictionary ret_dict and convert them to torch.FloatTensor and torch.IntTensor formats for the PyTorch model to handle

        model = self.models[gts['model_list'][idx]].astype(np.float32)  #Select the 3D model of the corresponding instance (usually a point cloud or model grid) from gts['model_list'] and convert it to type np.float32
        translation = gts['translations'][idx].astype(np.float32)       #The translation vector of the current instance (perhaps the object's position in 3D space) is obtained from gts['translations'], also converted to np.float32
        rotation = gts['rotations'][idx].astype(np.float32)             #Gets the rotation matrix of the current instance from gts['rotations'], converted to type np.float32
        size = gts['scales'][idx] * gts['sizes'][idx].astype(np.float32) #Get the dimensions of the instance from gts['scales'] and gts['sizes'] and calculate the final scale, usually the size of the object

        # symmetry
        if cat_id in self.sym_ids:  #If cat_id belongs to the symmetric object class (specified by self.sym_ids), the rotation matrix of the object is processed. The symmetry correction of the rotation matrix is calculated here
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                [0.0,            1.0,  0.0           ],
                                [theta_y/r_norm, 0.0,  theta_x/r_norm]])  #theta_x and theta_y are used to calculate the combined values of some elements in the rotation matrix and to normalize them (r_norm). Then, a symmetry mapping matrix s_map is generated, which is used to adjust the rotation matrix so that the symmetry of the object is corrected
            rotation = rotation @ s_map     #rotation Matrix is combined with s_map by matrix multiplication to complete symmetry adjustment

        qo = (pts - translation[np.newaxis, :]) / (np.linalg.norm(size)+1e-8) @ rotation  #pts is the three-dimensional point cloud of the object in the image. After subtracting the translation amount and applying the rotation matrix to the object size normalization (Np.Linalga.norm (size)), the three-dimensional point cloud qo (with actual position and attitude) is obtained.
        dis = np.linalg.norm(qo[:, np.newaxis, :] - model[np.newaxis, :, :], axis=2)  #dis Is a distance matrix representing the Euclidean distance between the object model and the target object (qo). Here np.linalg.norm is used to calculate the distance between two sets of points, i.e. the distance between each target point and the model point
        pc_mask = np.min(dis, axis=1)     #pc_mask Is the minimum distance between each target point and all model points. The purpose of this step is to find the target point closest to the model point and generate a marker (mask).
        pc_mask = (pc_mask < self.threshold)    #The minimum distance from each target point to the model point is compared with a predefined threshold (self.threshold) to generate a Boolean mask (pc_mask). The valid point mask value is True if the minimum distance is less than the threshold, False otherwise

        ret_dict['model'] = torch.FloatTensor(model)   # 3D Model point cloud
        ret_dict['qo'] = torch.FloatTensor(qo)        # 3D point cloud after rotation and shift
        ret_dict['translation_label'] = torch.FloatTensor(translation)  #The translation label of the object
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)    #The rotating label of the object
        ret_dict['size_label'] = torch.FloatTensor(size)            #Size label of the object
        ret_dict['pc_mask'] = torch.FloatTensor(pc_mask)            #Mask, indicating the valid point
        return ret_dict


class TestDataset():
    def __init__(self, image_size, sample_num, data_dir, setting, dataset_name):
        self.dataset_name = dataset_name
        assert dataset_name in ['camera', 'real']  #Check whether the data set type is camera or real
        self.data_dir = data_dir
        self.setting = setting
        self.img_size = image_size
        self.sample_num = sample_num
        if dataset_name == 'real':    #Set the intrinsics of the camera according to dataset_name and load the result_pkl_list in the corresponding path
            self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
            result_pkl_list = glob.glob(os.path.join(self.data_dir, 'segmentation_results', 'REAL275', 'results_*.pkl'))
        elif dataset_name == 'camera':
            self.intrinsics = [577.5, 577.5, 319.5, 239.5]
            result_pkl_list = glob.glob(os.path.join(self.data_dir, 'segmentation_results', 'CAMERA25', 'results_*.pkl'))
        self.result_pkl_list = sorted(result_pkl_list)
        n_image = len(result_pkl_list)
        print('no. of test images: {}\n'.format(n_image))

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])  #xmap and ymap are NumPy arrays representing image width and height coordinate grids, respectively, for subsequent depth map and point cloud coordinate calculations
        self.sym_ids = [0, 1, 3]    # 0-indexed Represents a class index of symmetric objects
        self.norm_scale = 1000.0    # normalization scale Used for standardization of depth maps
        self.transform = transforms.Compose([transforms.ToTensor(),   #Standard image preprocessing combination
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])  #The image is converted to Tensor and normalized

    def __len__(self):
        return len(self.result_pkl_list)        #Returns the number of data in the result_pkl_list, representing the number of samples in the test dataset

    def __getitem__(self, index):
        path = self.result_pkl_list[index]  #Obtain the path of the test sample from result_pkl_list using path

        with open(path, 'rb') as f:
            data = pickle.load(f)#Use cPickle.load() to load the data and get the prediction results (pred_data and pred_mask)
        image_path = os.path.join(self.data_dir, data['image_path'][5:])

        pred_data = data
        pred_mask = data['pred_masks']  #Extract relevant information from data, including image paths, depth maps, etc

        num_instance = len(pred_data['pred_class_ids'])
        # rgb
        rgb = cv2.imread(image_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3  Use OpenCV to read images (RGB order) and convert them to BGR order (image format commonly used for OpenCV processing)

        # pts
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        if self.dataset_name == 'real':
            depth = load_depth(image_path) #480*640
        else:
            depth = load_composed_depth(image_path) #Load the depth map based on the data set type, calling load_depth() if it is of type 'real' or load_composed_depth() otherwise.

        if depth is None:
            # random choose
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        depth = fill_missing(depth, self.norm_scale, 1)  #The depth map is then filled with missing values

        xmap = self.xmap
        ymap = self.ymap  #Use the depth map and camera parameters (cam_fx, cam_fy, cam_cx, cam_cy) to calculate point cloud coordinates in 3D space
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3 pts0, pts1, pts2 represent the x, y, and depth coordinates of the image respectively, which are finally merged into a 3D point cloud

        all_rgb = []
        all_pts = []
        all_cat_ids = []
        all_choose = []
        flag_instance = torch.zeros(num_instance) == 1 #Creates a list to store the final returned data, including RGB images, 3D point clouds, category ids, and selected point indexes

        for j in range(num_instance):   #For each instance, the prediction mask is extracted and the bounding box (rmin, rmax, cmin, cmax) is calculated, then the mask and the valid area of the depth map are filtered according to the bounding box
            inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
            rmin, rmax, cmin, cmax = get_bbox(pred_data['pred_bboxes'][j])
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose)>16:  #            if len(choose)>16:  #Randomly select sample_num points from the candidate points as target object samples, ensuring that the number of samples for each instance does not exceed the predetermined upper limit
                if len(choose) <= self.sample_num:
                    choose_idx = np.random.choice(len(choose), self.sample_num)
                else:
                    choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
                choose = choose[choose_idx]
                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]

                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                instance_rgb = cv2.resize(instance_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                instance_rgb = self.transform(np.array(instance_rgb))  #Crop the RGB image of the target instance according to the bounding box and resize it to img_size, then apply predefined transformations (such as convert to Tensor and normalization)
                crop_w = rmax - rmin
                ratio = self.img_size / crop_w
                col_idx = choose % crop_w
                row_idx = choose // crop_w
                choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

                cat_id = pred_data['pred_class_ids'][j] - 1 # convert to 0-indexed
                all_pts.append(torch.FloatTensor(instance_pts))
                all_rgb.append(torch.FloatTensor(instance_rgb))
                all_cat_ids.append(torch.IntTensor([cat_id]).long())
                all_choose.append(torch.IntTensor(choose).long())
                flag_instance[j] = 1   #Add the processed RGB image, 3D point cloud, category ID, and selected point index to the corresponding list and mark the instance as valid

        if len(all_pts) == 0:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)  #If there are no valid instances (that is, all_pts is empty), a different sample is randomly selected and __getitem__ is re-called

        ret_dict = {}
        ret_dict['pts'] = torch.stack(all_pts) # N*3
        ret_dict['rgb'] = torch.stack(all_rgb)
        ret_dict['choose'] = torch.stack(all_choose)
        ret_dict['category_label'] = torch.stack(all_cat_ids).squeeze(1)

        ret_dict['gt_class_ids'] = torch.tensor(data['gt_class_ids'])
        ret_dict['gt_bboxes'] = torch.tensor(data['gt_bboxes'])
        ret_dict['gt_RTs'] = torch.tensor(data['gt_RTs'])
        ret_dict['gt_scales'] = torch.tensor(data['gt_scales'])
        ret_dict['gt_handle_visibility'] = torch.tensor(data['gt_handle_visibility'])

        ret_dict['pred_class_ids'] = torch.tensor(pred_data['pred_class_ids'])[flag_instance==1]
        ret_dict['pred_bboxes'] = torch.tensor(pred_data['pred_bboxes'])[flag_instance==1]
        ret_dict['pred_scores'] = torch.tensor(pred_data['pred_scores'])[flag_instance==1]


        ret_dict['index'] = torch.IntTensor([index]) #eturn all processed data (RGB images, point clouds, category labels, bounding boxes, prediction results, etc.) in dictionary format. Each data item is processed in PyTorch Tensor format using torch.stack()
        return ret_dict