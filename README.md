### 基于 CNN 的验证码识别

#### 一、项目环境
##### 开发环境: centos7, python3.5.0
##### 依赖python库: keras.2.1.5, tensorflow.1.4.1, h5py.2.8.0, pillow.5.2.0

#### 二、脚本说明
##### 1.captcha_train.py 用于训练cnn模型
##### 2.captcha_evaluation.py 用于模型评估
##### 3.captcha_predict.py 用于模型预测 

#### 三、执行顺序
###### step_1. 先执行captcha_train.py 训练模型
###### step_2. 再执行captcha_evaluation.py 评估模型
###### step_3. 可调用 captcha_predict.py 进行模型预测

#### 四、模型结构
###### 启动训练时，模型结构会显示到控制台

#### 五、超参说明
| 超参数 | 参数说明 |
| ---- | ---- |
| FILTERS_1 | 第一层卷积核数目 |
| FILTERS_2 | 第二层卷积核数目 |
| FILTERS_3 | 第三层卷积核数目 |
| FILTERS_4 | 第四层卷积核数目 |
| KEEP_PROB | dropout节点比例 | 
| batch_size | 批量训练数据大小 |
| epochs | 迭代次数 |
| val_loss | 验证集上损失函数值 |
| val_acc | 验证集上模型准确率 | 

#### 六、模型评估
###### 因训练数据集需要自行采集，代码附带数据集仅供参照。具体模型评估需根据不同数据集而确定。

#### 七、References
##### [1] https://github.com/PatrickLib/captcha_recognize
