# 修改自Mask2Former
[Mask2Former源码](https://github.com/facebookresearch/Mask2Former)\
修改说明：
* 网络结构并未做任何改动，核心是去除对detectron2框架的依赖
* 修改数据加载方式，将源码中按迭代次数训练的方式修改为按epoch方式，默认训练300个epoch
* 数据增强修改成使用imgaug
* 只完成了语义分割的demo，未写实例分割代码
* 只训练了resnet50作为backbone，swin未做调试

# 网络结构
backbone: resnet50\
Decoder: DefomTransformer + CrossAtten + SelfAtten结构，参见Mask2Former源码中的的PixelDecoder + TransformerDecoder

# 运行环境
* 推理测试：内存16G以上；GPU显存4G以上
* 网络训练：根据图片尺寸大小和解码层数不定，当图片最大尺寸512且解码层数为4时推荐2张3090，batch_size=6
# 使用
1. 安装requirements.txt中的包，在ubuntu20.04下测试通过，理论上在windows下也没问题；
2. 下载模型[mask2former_resnet50](https://pan.baidu.com/s/16EsPxfn0L9ZoF-YtNY5KwA), 提取码：usyz
3. 将模型拷贝到项目的ckpt文件夹下
4. 将测试图片拷贝到test文件夹下，或者任意指定文件夹(如果是用户指定文件夹，将文件夹路径配置在Base-segmention.yaml中的TEST.TEST_DIR)，默认结果输出到output文件夹下，也可通过TEST.SAVE_DIR自行配置
5. 默认使用了detectron2中的visualizer类进行输出显示，如果不想安装detectron2或者安装后有问题，也可使用项目中的显示方式，与detectron2的区别在于没有对显示出类别名称，其余保持一致。对Segmentation.py进行如下修改：(1)注释掉17、18行和118行对detectron2包的引入和初始化；(2)放开第144行的显示调用；(3)注释掉145到147行即可
```python
    mask_img = self.postprocess(mask_img, transformer_info, (img_width, img_height))
    self.visualize.show_result(img, mask_img, output_path)
    # v = Visualizer(np.array(img), ade20k_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    # semantic_result = v.draw_sem_seg(mask_img).get_image()
    # cv2.imwrite(output_path, semantic_result)     
```
<div align="center">
  <img src="https://bowenc0221.github.io/images/maskformerv2_teaser.png" width="100%" height="100%"/>
</div><br/>

# 模型训练
1. 数据集准备，下载ADEChallengeData2016数据集，解压到你指定的文件夹下，比如：/home/xx/data
2. 配置Base-segmention.yaml文件，修改DATASETS.ROOT_DIR为第一步数据集所在文件夹，比如：ROOT_DIR: '/home/dataset/'
3. 多尺度训练配置，修改Base-segmention.yaml文件中INPUT.CROP.SIZE，INPUT.CROP.MAX_SIZE为训练时最大的图像尺寸，如果硬件受限可以调小
4. transformer解码层数配置，修改maskformer_ake150.yaml文件中的MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS，默认是4，论文源码中是6，可以根据硬件情况适当调小
5. 多卡训练,在main.py中指定显卡序号即可，执行：
```
python -m torch.distributed.launch --nproc_per_node=2 main.py；
```
6. 单卡训练，将mian.py中的序号改为0，然后注释掉maskformer_train.py中的第213行(用于平均所有GPU上的loss)，单卡训练会很慢，建议至少双卡
# 准备自己的数据集
1. 将image与label分别存入两个不同的文件夹下
2. 编写*.odgt文件，格式参考dataset/training.odgt
3. 修改配置文件中的DATASETS.TRAIN和DATASETS.VALID分别为自定义文件路径，修改ROOT_DIR为image和label文件上级目录的父文件夹，同时修改PIXEL_MEAN和PIXEL_STD为自定义数据集的均值方差
4. 修改maskformer_ake150.yaml中的MODEL.SEM_SEG_HEAD.NUM_CLASSES为自定义数据集的类别数，此处不包含背景类
