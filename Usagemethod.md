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
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen/Qwen2___5-VL-7B-Instruct \   # 指定模型文件路径
    --port 1238 \                                               # API 服务端口
    --gpu-memory-utilization 0.7 \                               # GPU 显存占用比例
    --max-num-seqs 64                                           # 最大并行序列数
```


###  3. 启动程序
```bash
python batch_processor.py /workspace/input_path  ~/workspace/output/result-7b-5 --config config/config.yaml --log-level DEBUG
```