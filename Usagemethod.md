## 使用说明

方式：需要开启两个终端，终端1启动模型服务，终端2启动程序

### 1. 启动 Docker 容器

在两个终端内运行：

```bash
docker run --runtime=nvidia -it --rm --network=host \
    -v /nfs/samba/models/HJJ-VQA-data/input_path:/workspace/input_path \   # 映射测试数据
    -v ~/repo/hjj:/workspace/output \                                    # 映射输出
    -v /nfs/samba/models/vlm/Qwen_Qwen2.5-VL-7B-Instruct:/workspace/models \  # 映射模型文件
    llm-vqa:latest
```


###  2. 启动模型服务
```bash
export CUDA_VISIBLE_DEVICES=1
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen/Qwen2___5-VL-7B-Instruct \   # 指定模型文件路径
    --server-name qwen-vl                                       #model-name
    --port 1238 \                                               # API 服务端口
    --gpu-memory-utilization 0.8 \                               # GPU 显存占用比例
    --max-num-seqs 64       \                                    # 最大并行序列数
    --tensor-parallel-size 1 \ 
```


###  3. 启动程序
```bash
python batch_processor.py /workspace/input_path  ~/workspace/output/result-7b-5 --config config/config.yaml --log-level DEBUG

#使用说明：
# /workspace/input_path：代表输入地址，一般不用改
# ~/workspace/output/result-7b-5：代表输出地址，主要修改-5这个序号，跑一次注意加一，否则会有预处理超时。
```

### 4. 调试config
```yaml
# 批量处理配置
batch_processing:
  max_concurrent_requests: 8  # 最大并发请求数，如果速度比较慢可以调大，但是如果出现out of memory，调小
  request_interval: 3  # 请求间隔（秒）同上，但是是相反调整

# 图像预处理配置
image_preprocessing:
  enabled: true  # 是否启用图像预处理
  max_resolution: [1080,1080]  # 最大分辨率 [width, height]，根据比赛方图片分辨率调整，如果是几万*几万就调为2048*2048。小的话就不动。
  # TIF图像处理配置
  tif_processing_enabled: true  # 是否启用TIF处理
  target_tile_size: 1080  # 切片目标尺寸，同max_resolution
optimization:
  batch_size: 5  # 每批次处理的查询数量,可以调整大
# 日志配置
logging:
  file: "/workspace/output/logs/app.log"  # 日志文件路径