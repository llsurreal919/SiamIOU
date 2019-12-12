# SiamIOU Tracker
Python implementation of the SiamIOU Tracker.

A multiple vehicle tracking approach is designed to effectively integrate SOT based forward position prediction with [IOU Tracker](https://github.com/bochinski/iou-tracker) to enhance the detection results in the association phase and consequently improve the tracking performance. The  proposed method outperforms the state-of-the-art methods on the [UA-DETRAC](https://detrac-db.rit.albany.edu/) dataset while running at a real-time speed.


The SOT tracker used in this method is [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html), which is a high-quality, high-performance algorithm for visual tracking. We make use of [pysot](https://github.com/STVIR/pysot), which is a software system designed by SenseTime Video Intelligence Research team, to implement it.

# User Guide for the code
## Preparation
1) Clone the repository
`git clone https://github.com/llsurreal919/SiamIoU.git`   
2) Configure the environment according to pysot instruction.
3) The code is prepared for a environment with Python==3.7 and Pytorch==0.4 with the required CUDA and CudNN library.   
4) Other packages can be installed by:   
`sudo pip install -r requirements.txt` 
## Download testing datasets
Download datasets and put them into current directory. Datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F), and the EB detections also are included.
## Running the tracker
1)Configure the path of the file and call the main script. You can configure the config.yaml and model.pth by refering to pysot's instructions.
2)Call the main script.
```bash
python main.py \
    --config path/to/config.yaml \
    --snapshot path/to/model.pth \
    --dataset UA-DETRAC
```
## Results from DETRAC Dataset
To reproduce the reported results, download and extract the [DETRAC-toolkit](http://detrac-db.rit.albany.edu/download), and the detections you want to evaluate.
Read the txt file of tracking results from tracker into DETRAC-toolkit in turn.
You should obtain something like the following results for the 'DETRAC-Train' set:

### DETRAC-Test (Overall) Results
The reference results are taken from the [UA-DETRAC results](http://detrac-db.rit.albany.edu/TraRet) site.

| Tracker       | Detector | PR-MOTA | PR-MOTP   | PR-MT     | PR-ML     | PR-IDs   | PR-FP      | PR-FN      | Speed          |
| ------------- | -------- | ------- | ----------| --------- | --------- | -------- | ---------- | ---------- | -------------- |
|DAN            | EB       | 20.2\%  |26.3\%     |14.5\%     |18.1\%     |518.2     |9747.8      |135978.1    |6.3 fps         |
|CMOT           | CompACT  | 12.6\%  |36.1\%     |16.1\%     |18.6\%     |285.3     |57885.9     |167110.8    |3.79 fps        |
|DCT            | R-CNN    | 11.7\%  |38.0\%     |10.1\%     |22.8\%     |758.7     |336561.2    |210855.6    |0.71 fps        |
|H<sup>2</sup>T | CompACT  | 12.4\%  |35.7\%     |14.8\%     |19.4\%     |852.2     |51765.7     |173899.8    |3.02 fps        |
|IHTLS          | CompACT  | 11.1\%  |36.8\%     |13.8\%     |19.9\%     |953.6     |53922.3     |180422.3    |19.79 fps       |
|IOU            | R-CNN    |16.0\%   |38.3\%     |13.8\%     |20.7\%     |5029.4    |22535.1     |193041.9    |100,840 fps     |
|IOU            | EB       |19.4\%   |28.9\%     |17.7\%     |18.4\%     |2311.3    |14796.5     |171806.8    |6,902 fps       |
|SiamIOU        | EB       |21.5\%   | 28.6\%    |23.0\%     |19.6\%     |479.9     |21137.8     |169095.0    |20.1fps         |

#### EB detections
We obtained our copy of detections from the authors of the original [IOU Tracker](https://github.com/bochinski/iou-tracker).



