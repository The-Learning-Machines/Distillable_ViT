# Distillable_ViT
A repository for the Distillable ViT, with and without Label Smoothing

## Distillation

A recent <a href="https://arxiv.org/abs/2012.12877">paper</a> has shown that use of a distillation token for distilling knowledge from convolutional nets to vision transformer can yield small and efficient vision transformers. This repository offers the means to do distillation easily.

ex. distilling from Resnet50 (or any teacher) to a vision transformer

## Usage of LSR (Label Smoothing)

```python
!pip install timm==0.3.2
from timm.loss import LabelSmoothingCrossEntropy

value = 0.1 #check which is fine
smoothing = True

if smoothing : 
    base_criterion = LabelSmoothingCrossEntropy(smoothing = value)
else : 
    base_criterion = nn.CrossEntropyLoss()

criterion = DistillationLoss(
    base_criterion, teacher, 'none', 0.5, 1.0
)  #the args are distillation-type, distillation-alpha and distillation-tau, type = choices=['none', 'soft', 'hard']
```
## TODO

Learning with Retrospection (LWR) - https://arxiv.org/abs/2012.13098

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
