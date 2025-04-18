# Graph-OMO

Introduce Graph Convolution Network to OMOMO's 1st stage network[1] to make it possible to process sequence with variable amount of objects. We use himo dataset[2] for hands manipulation sequence generation of multiple objects  
The detailed explanation of Graph-OMO can be seen in this [report](https://github.com/Hongboooooo/Graph-OMO/blob/main/PracticalReport_Hongbo.pdf)  
The architecture of Graph-OMO: 
>  ![image](https://github.com/Hongboooooo/Graph-OMO/blob/main/GOMO_Pipeline.png)

The Figure below shows the improvements brought by Graph-OMO and Attention Aggregation:  
>  ![image](https://github.com/Hongboooooo/Graph-OMO/blob/main/Graph-OMO_Curves_3.png)

Visual example:
> Two green dots in the gif below are the inferred result of OMO  
> Yellow skeleton is the ground truth
![image](https://github.com/Hongboooooo/Graph-OMO/blob/main/omo-himo.gif)

Reference:  
> [1] https://github.com/lijiaman/omomo_release  
> [2] https://github.com/LvXinTao/HIMO_dataset


