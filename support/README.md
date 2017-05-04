# Pulmonary nodules detection
Pulmonary nodules detection play a significant role in the early detection and treat-  
ment of lung cancer. And false positive reduction is the one of the major parts of  
pulmonary nodules detection systems. In this study a novel method aimed at rec-  
ognizing real pulmonary nodule among a large group of candidates was proposed.  
The method consists of three steps: appropriate receptive field selection, feature  
extraction and a strategy for high level feature fusion and classification. Receptive  
field’s objective is to fit tradeoff between covering 3D nature of nodules appearance  
and parameters amount. Due to that three parallelepipeds with it’s largest base  
parallel along to the frontal, horizontal and side views respectively. Deep residual  
3D CNN acing over such receptive fields prior to fusion part provide an opportu-  
nity for spatial merging. Multi-scale information was handled by dimensionality  
reduction step as part of feature extraction network. The dataset consists of 888  
patient’s chest volume low dose computer tomography (LDCT) scans, selected from  
publicly available LIDC-IDRI dataset. This dataset was marked by LUNA16 chal-  
lenge organizers resulting in 1186 nodules. Trivial data augmentation and dropout  
were applied in order to avoid overfitting. Our method achieved high competition  
performance metric (CPM) of 0.735 and sensitivities of 78.8% and 83.9% at 1 and  
4 false positives per scan, respectively. This study is also accompanied by detailed  
descriptions and results overview in comparison with the state of the art solutions.

