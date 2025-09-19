# 比赛用统一启动脚本使用说明

## 概述

本脚本为比赛专用的统一启动脚本，集成了vLLM服务启动和批量处理功能。一个脚本完成所有操作，满足比赛要求。

## 使用方法

### 基本用法

```bash
./use.sh <输入文件路径> <输出文件路径>
```

### 参数说明

- `输入文件路径`: 待处理的输入数据路径
- `输出文件路径`: 处理结果的输出路径

### 示例

```bash
# 处理输入数据并输出到指定位置
./use.sh /workspace/input_path /workspace/output/result-7b-7

# 或者使用完整路径
./use.sh /home/user/input /home/user/output/results
```

## 脚本功能

1. **自动启动vLLM服务**: 根据配置文件自动启动vLLM OpenAI API服务器
2. **健康检查**: 等待服务启动完成并进行健康检查
3. **批量处理**: 运行批量处理器处理输入数据
4. **自动清理**: 处理完成后自动关闭vLLM服务

## 配置文件

配置文件位于 `config/config.yaml`，包含以下vLLM服务配置：

```yaml
vllm_server:
  model_path: "/workspace/models/Qwen_Qwen2.5-VL-7B-Instruct/Qwen/Qwen2___5-VL-7B-Instruct"
  served_model_name: "qwen-vl"
  port: 1238
  gpu_memory_utilization: 0.8
  tensor_parallel_size: 1
  cuda_visible_devices: "4"
```

### 配置参数说明

- `model_path`: 模型文件路径
- `served_model_name`: 服务中的模型名称
- `port`: API服务端口
- `gpu_memory_utilization`: GPU内存使用率
- `tensor_parallel_size`: 张量并行大小
- `cuda_visible_devices`: 可见的CUDA设备编号

## 日志输出

脚本运行时会输出详细的日志信息，包括：

- vLLM服务启动状态
- 服务健康检查结果
- 批量处理进度
- 错误信息和调试信息

## 注意事项

1. **环境要求**: 确保已安装vLLM和相关依赖
2. **模型路径**: 确保配置文件中的模型路径正确
3. **端口占用**: 确保配置的端口未被占用
4. **GPU资源**: 确保有足够的GPU内存
5. **权限**: 确保脚本有执行权限 (`chmod +x use.sh`)

## 错误排查

### 常见问题

1. **服务启动失败**
   - 检查模型路径是否正确
   - 检查GPU内存是否足够
   - 检查端口是否被占用

2. **处理失败**
   - 检查输入路径是否存在
   - 检查输出目录是否有写权限
   - 查看详细日志信息

3. **权限问题**
   ```bash
   chmod +x use.sh
   ```

## 技术细节

脚本内部流程：

1. 解析命令行参数
2. 加载配置文件
3. 设置CUDA环境变量
4. 启动vLLM服务器进程
5. 等待服务启动完成
6. 运行批量处理器
7. 自动清理和关闭服务

## 支持

如有问题，请检查：
1. 配置文件格式是否正确
2. 模型和输入文件是否存在
3. 系统资源是否充足
4. 日志输出中的错误信息