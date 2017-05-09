# Pulmonary nodules detection
Pulmonary nodules detection plays a significant role in the early detection and treatment  
of lung cancer. And false positive reduction is the one of the major parts of pulmonary   
nodules detection systems. In this study was provided a framwork that solves following problems:  
lungs segmentation, left and right lung separation, nodule candidates detection and false   
positive reduction. Novel methods was proposed aimed at lungs separation and recognizing real  
pulmonary nodule among a large group of candidates was proposed. The first algorithm via    
dilation propagation approach, described in this work, gains better performance in term of   
time complexity in comparison with the State of the Art methods. The latter algorithm consists   
of three steps: appropriate receptive field selection, feature extraction and a strategy for   
high level feature fusion and classification. Receptive field's objective is to fit tradeoff   
between covering 3D nature of nodules appearance and parameters amount. Deep residual 3D CNN   
acing over such receptive fields prior to fusion part provide an opportunity for spatial merging.   
Multi-scale information was handled by dimensionality reduction step as part of feature extraction   
network. The dataset consists of 888 patient's chest volume low dose computer tomography (LDCT)  
scans, selected from publicly available LIDC-IDRI dataset. This dataset was marked by LUNA16   
challenge organizers resulting in 1186 nodules. Trivial data augmentation and dropout were applied   
in order to avoid overfitting. Proposed method achieved high competition performance metric (CPM)  
of 0.735  and sensitivities of 78.8\% and 83.9\% at 1 and 4 false positives per scan, respectively.    
This study is also accompanied by detailed descriptions and results overview in comparison with the 
state of the art solutions.


### Prerequarements ###

* Numpy, Scipy, Scikit-Image
* SimpleITK, PyDICOM
* Keras, TensorFlow >= 0.10, GPU supported


### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* a.dobrenkiy@innopolis.ru

