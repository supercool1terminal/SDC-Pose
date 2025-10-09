# KA-Pose
![image](./fig1.png)
## Environment Settings
The code has been tested with

- python 3.9
- torch 1.12
- cuda 11.3

Some dependencies:
```
pip install gorilla-core==0.2.5.3
pip install opencv-python

cd model/pointnet2
python setup.py install
```
## Data Processing
### NOCS dataset
- Download and preprocess the dataset following [DPDN](https://github.com/JiehongLin/Self-DPDN)
- Download and unzip the segmentation results [here](http://home.ustc.edu.cn/~llinxiao/segmentation_results.zip)

Put them under ```PROJ_DIR/data```and the final file structure is as follows:
```
data
├── camera
│   ├── train
│   ├── val
│   ├── train_list_all.txt
│   ├── train_list.txt
│   ├── val_list_all.txt
├── real
│   ├── train
│   ├── test
│   ├── train_list.txt
│   ├── train_list_all.txt
│   └── test_list_all.txt
├── segmentation_results
│   ├── CAMERA25
│   └── REAL275
├── camera_full_depths
├── gts
└── obj_models
```
### HouseCat6D
Download and unzip the dataset from [HouseCat6D](https://sites.google.com/view/housecat6d) and the final file structure is as follows:
```
HOUSECAT6D_DIR
├── scene**
├── val_scene*
├── test_scene*
└── obj_models_small_size_final
```
## Train
### Training on NOCS
```
python train.py --config config/REAL/camera_real.yaml
```
### Training on HouseCat6D
```
python train_housecat6d.py --config config/HouseCat6D/housecat6d.yaml
```

## Evaluate 
- Evaluate on NOCS:
```
python test.py --config config/REAL/camera_real.yaml --test_epoch 30
```
- Evaluate on HouseCat6D:
```
python test_housecat6d.py --config config/HouseCat6D/housecat6d.yaml --test_epoch 150
```
## Visualization
For visualization, please run
```
python visualize.py --config config/REAL/camera_real.yaml --test_epoch 30
```
![image](./Vis.png)
