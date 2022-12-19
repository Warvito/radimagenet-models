# Optional list of dependencies required by the package
dependencies = ["torch"]

from radimagenet_models.models.densenet import densenet121
from radimagenet_models.models.inception import inception_v3
from radimagenet_models.models.resnet import resnet50