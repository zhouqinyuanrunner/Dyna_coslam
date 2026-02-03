# Neural Real-Time RGB-D SLAM in Dynamic Environments




## Installation

Please follow the instructions below to install the repo and dependencies.
I recommend to install it in 133.2.208.229

```bash
Please download '/mnt/new_kraz/students/2024/users/zhou.qinyuan/nerual_slam/coslam3_image.tar' to your own workspace
docker run -it --name coslam3v_new --shm-size=5G coslam3_image bash
rm -rf Co-SLAM
git clone http://jupiter.vssad.it.aoyama.ac.jp/zhou.qinyuan/dynaslam.git
mv dynaslam Co-SLAM
cd Co-SLAM
conda activate coslam
pip install open3d
mkdir data
cd data
mkdir scannet_d
```


## Dataset

#### Test data
The test data is in smb://133.2.208.200/students/2024/users/zhou.qinyuan/nerual_slam/scannet_d/scene0000_00_d/
Please download it and put it in /Co-SLAM/data/scannet_d
```bash
docker cp /mnt/new_kraz/students/2024/users/zhou.qinyuan/n
erual_slam/scannet_d/scene0000_00_d coslam3v_new://Co-SLAM/data/scannet_d
```

## Run and Evaluation

You can run Co-SLAM using the code below:

```
cd /Co-SLAM/
python coslam.py --config './configs/ScanNet/scene0000_d.yaml' 
```








