# darknet-ameliorate
This project is mainly based on the original darknet (well known for yolo/yolov2/yolov3, https://github.com/pjreddie/darknet), and devoted to be a better darknet, especially for yolo.

## code-references
- (yolo/yolov2/**yolov3**) https://github.com/pjreddie/darknet
- (**yolov4**) https://github.com/AlexeyAB/darknet
- (**Channel Slimming**) https://github.com/Eric-mingjie/network-slimming (https://github.com/liuzhuang13/slimming)
- (**Quantization, Google 2018**) https://arxiv.org/pdf/1806.08342.pdf
- **Others**

## features preview
- Data Augmentation
    - [x] rotation in rbox
    - [x] mosaic
- Model compression and acceleration 
    - [x] channel slimming (2020/7/11)
    - [x] knowledge distillation (2020/7/11)
    - [x] mutual learning (2020/7/11)
    - [x] post training quantization (2020/8/5)
    - [x] quantization aware training (2020/10/26)
    - [ ] keypoint detection (updating)
