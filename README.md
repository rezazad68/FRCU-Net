# [Deep Frequency Re-calibration U-Net for Medical Image Segmentation](https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/papers/Azad_Deep_Frequency_Re-Calibration_U-Net_for_Medical_Image_Segmentation_ICCVW_2021_paper.pdf)

Deep frequency re-calibration U-Net (FRCU-Net) for medical image segmentation. This method aims to represent an object in
terms of frequency to reduce the effect of texture bias, consequntly resultign in a better generalization performance. Following approach implements the idea of Laplacian pyramid in the bottleneck layer of the U-shaped structure and adaptively re-calibrate the frequency representations to encode shape and texture information. The method is evaluated on five datasets ISIC 2017, ISIC 2018, the PH2, lung segmentation, and SegPC 2021 challenge datasets. If this code helps with your research please consider citing the following paper:
</br>

> [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [Afshin Bozorgpour](https://scholar.google.ae/citations?user=OUZkcNsAAAAJ&hl=vi), [M. Asadi](https://scholar.google.com/citations?hl=en&user=8UqpIK8AAAAJ&view_op=list_works&sortby=pubdate),  [Dorit Merhof](https://scholar.google.de/citations?user=JH5HObAAAAAJ&hl=de) and [Sergio Escalera](https://scholar.google.com/citations?hl=en&user=oI6AIkMAAAAJ&view_op=list_works&sortby=pubdate) "Deep Frequency Re-calibration U-Net for Medical Image Segmentation", ICCV, 2021, download [link](https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/papers/Azad_Deep_Frequency_Re-Calibration_U-Net_for_Medical_Image_Segmentation_ICCVW_2021_paper.pdf).

#### Please consider starring us, if you found it useful. Thanks

## Updates
- October 10, 2021: Initial release of the code along with trained weights for Skin lesion segmentation on ISIC 2017, ISIC 2018 and PH2. 

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3 </br>
- Keras 2.2.0 </br>
- tensorflow 1.13.1 </br>


## Run Demo
For training deep model and evaluating on each data set follow the bellow steps:</br>
1- Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `Train_Skin_Lesion_Segmentation.py` for training the model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. </br>
4- For performance calculation and producing segmentation result, run `Evaluate_Skin.py`. It will represent performance measures and will saves related results in `output` folder.</br>

**Notice:**
For training and evaluating on ISIC 2017 and ph2 follow the bellow steps: :</br>
**ISIC 2017**- Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18\7`. </br> then Run ` 	Prepare_ISIC2017.py` for data preperation and dividing data to train,validation and test sets. </br>
**ph2**- Download the ph2 dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract it then Run ` 	Prepare_ph2.py` for data preperation and dividing data to train,validation and test sets. </br>
Follow step 3 and 4 for model traing and performance estimation. For ph2 dataset you need to first train the model with ISIC 2018 data set and then fine-tune the trained model using ph2 dataset.



## Quick Overview
### Diagram of the proposed method
![Diagram of the proposed Attention mechanism](https://github.com/rezazad68/FRCU-Net/blob/main/Figures/proposed_method.png)

### Frequency attention mechanism
![Diagram of the proposed Attention mechanism](https://github.com/rezazad68/FRCU-Net/blob/main/Figures/attention.png)



#### Performance Evalution on the Skin Lesion Segmentation ISIC 2018

Methods | Year |F1-scores | Sensivity| Specificaty| Accuracy | PC | JS 
------------ | -------------|----|-----------------|----|---- |---- |---- 
Ronneberger and etc. all [U-net](https://arxiv.org/abs/1505.04597)	     	    |2015   | 0.647	|0.708	  |0.964	  |0.890  |0.779 |0.549
Alom  et. all [Recurrent Residual U-net](https://arxiv.org/abs/1802.06955)	|2018	  | 0.679 |0.792 |0.928 |0.880	  |0.741	  |0.581
Oktay  et. all [Attention U-net](https://arxiv.org/abs/1804.03999)	|2018	  | 0.665	|0.717	  |0.967	  |0.897	  |0.787 | 0.566 
Alom  et. all [R2U-Net](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf)	        |2018	  | 0.691	|0.726	  |0.971	  |0.904	  |0.822 | 0.592
Azad et. all [BCDU-Net](https://github.com/rezazad68/LSTM-U-net/edit/master/README.md)	  |2019 	| 0.847	|0.783	  |0.980	  |0.936	  |0.922| 0.936
Asadi et. all [MCGU-Net](https://128.84.21.199/pdf/2003.05056.pdf)	  |2020	| 0.895	|0.848	  |0.986	  |0.955	  |0.947| 0.955
Azad et. all [Attention Deeplabv3p](https://www.bioimagecomputing.com/program/selected-contributions/)	  |2021	| **0.927**	|**0.915**	  |**0.986**	  |**0.973**	  |..| **0.973**


### Segmentation visualization
![ISIC 2018](https://github.com/rezazad68/FRCU-Net/blob/main/Figures/results_isic18.png)




### Model weights
You can download the learned weights for each dataset in the following table. 

Dataset |Learned weights
------------ | -------------
[ISIC 2018](http://www.isi.uu.nl/Research/Databases/DRIVE/) |[Deeplabv3pa](https://drive.google.com/file/d/10S9ewav837izWaraOlUB8OOQoWY9szzU/view?usp=sharing)
[ISIC 2017](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) |[Deeplabv3pa](https://drive.google.com/file/d/1hXy-gKCHIG8myY9R4lB6GYE7xxbqM_Hj/view?usp=sharing)
[Ph2](https://www.kaggle.com/kmader/finding-lungs-in-ct-data/data) | [Deeplabv3pa](https://drive.google.com/file/d/1Ni9PldLL9bMYlyjcRxgDitr-MR6o-RY4/view?usp=sharing)

### Query
All implementation done by Reza Azad. For any query please contact us for more information.

```python
rezazad68@gmail.com

```


