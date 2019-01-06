*building detector, some stuff in segmentation
___
![](misc/both.png)

What we have:
 - torch backend with Adam optimizer
 - using resnet34 backbone plus unet decoder
 - train on 256, 512, 1024 patches, sequentialy in order to avoid BN problems
 - soft dice loss

---
Dependencies:
- `torch`
- `torchvision`
- `sklearn`
- `cv2`
- `numpy`
- `skimage`
---
Usage:
```bash
python train_resnet.py
```
For inference examples see `test model.ipynb`

TODO:
- try deeper model plus channel and spatial squeeze and excitation stuff from `say.py` - [ ]
- try add some differen level of dilation (aka atrous conv) in center of the model - [ ]
