# Distillable_ViT
A repository for the Distillable ViT, with and without Label Smoothing

Weights can be downloaded from - https://drive.google.com/drive/folders/1iGew6DMDdIorm-f_73fvL8sYpMIiwIw6?usp=sharing

## Distillation

A recent <a href="https://arxiv.org/abs/2012.12877">paper</a> has shown that use of a distillation token for distilling knowledge from convolutional nets to vision transformer can yield small and efficient vision transformers. This repository offers the means to do distillation easily.

ex. distilling from Resnet50 (or any teacher) to a vision transformer

## Usage of LSR (Label Smoothing)

```python
!pip install timm==0.3.2
from timm.loss import LabelSmoothingCrossEntropy

smoothing = True
retrospect = False

value = 0.1 
if smoothing : 
    base_criterion = LabelSmoothingCrossEntropy(smoothing = value)
    
elif retrospect: 
    base_criterion = LWR(
    k=1,
    update_rate=0.9,
    num_batches_per_epoch=len(train_data) // batch_size,
    dataset_length=len(train_data),
    output_shape=(2, ),
    tau=5,
    max_epochs=20,
    softmax_dim=1
)
      
else : 
    base_criterion = nn.CrossEntropyLoss()

criterion = DistillationLoss(
    base_criterion, teacher, 'none', 0.5, 1.0, smoothing, retrospect)
```
## Usage of Learning with Retrospection (LWR)

Refer to - https://github.com/The-Learning-Machines/LearningWithRetrospection/blob/main/LearningWithRetrospection.py

```python
smoothing = False
retrospect = True

value = 0.1 
if smoothing : 
    base_criterion = LabelSmoothingCrossEntropy(smoothing = value)
    
elif retrospect: 
    base_criterion = LWR(
    k=1,
    update_rate=0.9,
    num_batches_per_epoch=len(train_data) // batch_size,
    dataset_length=len(train_data),
    output_shape=(2, ),
    tau=5,
    max_epochs=20,
    softmax_dim=1
)
      
else : 
    base_criterion = nn.CrossEntropyLoss()

criterion = DistillationLoss(
    base_criterion, teacher, 'none', 0.5, 1.0, smoothing, retrospect)
```

## Citations

Code base from - https://github.com/lucidrains/vit-pytorch

```bibtex
@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{touvron2020deit,
  title={Training data-efficient image transformers & distillation through attention},
  author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv\'e J\'egou},
  journal={arXiv preprint arXiv:2012.12877},
  year={2020}
}
```
