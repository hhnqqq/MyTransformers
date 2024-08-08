## 使用方法：
- dp_train为序列并行和数据并行的文件，是否使用序列并行由num-sp-stages控制
- pp_train为流水线并行的文件
- trainer为训练器的代码，使用实例详见文件中的测试示例

## 关于注册器
训练代码会大量使用到注册器，使用说明如下：
- 使用数据并行或序列并行训练时：
    1. 首先使用注册器获得tokenizer（与model_name和tokenizer_name参数有关）
    2. 使用注册器获得模型config （与model_name和variant参数有关）
    3. 使用注册器获得模型类 （与model_name参数有关）
    4. 使用注册器获得模型训练类，即TrainModel类 （与model_name参数有关）
- 使用流水线并行训练时：
    4. 使用注册器获得流水线训练类，即PipelineModule类