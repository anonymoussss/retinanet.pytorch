# retinanet.pytorch
retinanet by pytorch.

Most of the code in this repository comes from  https://github.com/kuangliu/pytorch-retinanet/issues . I use kuangliu's repository and fix some bugs.

## Usage

- If you want to use the pretrained backbone , I only implemented R50 and R101 here. Please download the pretrained model by the following link.

  > 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  > 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

  after downloading the .pth file, put it into **./model** directory.

- Change the dataset path in the code.  I use voc12 in my experiment.

- Use the command below.

  ```
  python train.py --pretrained res50
  ```

- Test.

  ```
  python test.py
  ```

  