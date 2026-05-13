# 实验二相关材料
## 参考：
[香橙派 AIpro开发体验：使用YOLOV8对USB摄像头画面进行目标检测](https://blog.mvui.cn/detail/116256.html#yolov8_114)
**注意：在目前版本下以下代码会报错:**
```python
from ultralytics.utils import yaml_load
(code)
CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
```
，解决方法是替换成:
```python
from ultralytics.utils import YAML
(code)
CLASSES = YAML.load(check_yaml('coco128.yaml'))['names']
```

- video.ipynb:根据博客教程使用torch库在香橙派上对USB摄像头和小车画面进行目标检测的示例代码。(做yaml_load修复并增加小车画面检测部分)   