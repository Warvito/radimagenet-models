# pytorch_radimagenet_models
Unofficial support to RadImageNet pretrained models for Pytorch

(Download link generated with https://sites.google.com/site/gdocs2direct/?pli=1)

To download the model use:

```
from radimagenet_models.models.resnet import radimagenet_resnet50
resnet = radimagenet_resnet50()
```

or

```
import torch
model = torch.hub.load("Warvito/radimagenet-models", 'radimagenet_resnet50')
```
