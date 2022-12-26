### This is an implementation of the paper Exploring Task Structure for Brain Tumor Segmentation From Multi-Modality MR Images published on IEEE Transaction on Image Processing.  
### Environment
---
python(Tested on 3.6)  
TensoFlow(Tested on 1.8)  
### Data Set
---  
BraTS 2017  data set(including training set and validation set).   
Sample list file `*.txt`(eg. train.txt), which contains the filenames of the inputs.
### Test
---
Modify the configuration file `netetw.conf` and run:  
``` 
python etwtest.py
```
### Train
---  
The training of this implementation is divided into two stages.  
* The first stage is to train the three subnets respectively using the corresponding segmentation label.   
Modify the configuration file `net_type.conf` and  `netet.conf` for training `et` net.   
Modify the configuration file `net_type.conf` and `nettc.conf` for training `tc` net.   
Modify the configuration file `net_type.conf` and `netwt.conf` for training `wt` net.  
After Modifying the corresponding files, run the command below respectively:
``` 
python train.py
```
* The second stage is to train the combined net `etw` based on the configuration file `netetw.conf` by running:
```
python etwtrain.py
```