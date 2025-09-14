# 优化指南 - 大规模图像处理

## 🚀 优化版批量处理器特性

### 核心优化策略

1. **分批处理** - 将大量查询分成小批次，避免内存爆炸
2. **资源监控** - 实时监控内存使用，自动暂停和清理
3. **渐进式图像预处理** - 避免同时处理过多大图像
4. **智能缓存管理** - 定期清理缓存，防止内存泄漏
5. **错误恢复** - 检测到内存/显存问题时自动重试

### 使用方法

#### 1. 优化模式（推荐）
```bash
# 使用优化模式处理大规模数据
python batch_processor.py /path/to/input_data /path/to/output_path

# 指定批次大小
python batch_processor.py /path/to/input_data /path/to/output_path --batch-size 5
```

#### 2. 标准模式（小规模数据）
```bash
# 禁用优化，使用传统并发处理
python batch_processor.py /path/to/input_data /path/to/output_path --no-optimization
```

#### 3. 监控模式
```bash
# 启用详细日志监控资源使用
python batch_processor.py /path/to/input_data /path/to/output_path --log-level DEBUG --log-file logs/optimization.log
```

## ⚙️ 配置参数

### 优化配置 (config/config.yaml)

```yaml
# 优化配置
optimization:
  batch_size: 3                    # 每批次处理的查询数量
  memory_cleanup_interval: 5       # 每N个批次执行内存清理
  progressive_processing: true     # 渐进式处理
  max_memory_usage: 75.0          # 最大内存使用率(%)
  image_batch_size: 2             # 图像预处理批次大小

# 批量处理配置
batch_processing:
  max_concurrent_requests: 1       # 并发数量（建议保持1）
  request_interval: 5              # 请求间隔（秒）
  retry_attempts: 3               # 重试次数
  timeout: 500                    # 请求超时时间（秒）

# 图像预处理配置
image_preprocessing:
  enabled: true
  max_resolution: [2048, 2048]    # 最大分辨率
  quality: 85                     # JPEG质量
  parallel_workers: 4             # 预处理并行数
  tif_processing_enabled: true    # 启用TIF处理
  target_tile_size: 2048         # TIF切片尺寸
  grid_rows: 3                   # 网格行数
  grid_cols: 3                   # 网格列数
```

## 📊 性能对比

| 模式 | 内存使用 | 处理速度 | 稳定性 | 适用场景 |
|------|----------|----------|--------|----------|
| 优化模式 | 低且稳定 | 中等 | 高 | 大规模数据(>100张图) |
| 标准模式 | 高且波动 | 快 | 中等 | 小规模数据(<50张图) |

## 🔧 调优建议

### 根据数据规模调整

**小规模 (< 50张图像)**
```yaml
optimization:
  batch_size: 5
  memory_cleanup_interval: 10
  image_batch_size: 3
```

**中等规模 (50-200张图像)**
```yaml
optimization:
  batch_size: 3
  memory_cleanup_interval: 5
  image_batch_size: 2
```

**大规模 (> 200张图像)**
```yaml
optimization:
  batch_size: 2
  memory_cleanup_interval: 3
  image_batch_size: 1
```

### 根据服务器配置调整

**高配置服务器 (32GB+ RAM, 高端GPU)**
```yaml
optimization:
  batch_size: 5
  max_memory_usage: 85.0
  image_batch_size: 3
batch_processing:
  max_concurrent_requests: 2
```

**中等配置服务器 (16GB RAM, 中端GPU)**
```yaml
optimization:
  batch_size: 3
  max_memory_usage: 75.0
  image_batch_size: 2
batch_processing:
  max_concurrent_requests: 1
```

**低配置服务器 (8GB RAM, 入门GPU)**
```yaml
optimization:
  batch_size: 1
  max_memory_usage: 70.0
  image_batch_size: 1
batch_processing:
  max_concurrent_requests: 1
  request_interval: 10
```

## 🚨 故障排除

### 常见问题

**1. 内存不足错误**
```
解决方案：
- 减小 batch_size
- 增加 request_interval
- 启用 progressive_processing
```

**2. 显存爆炸**
```
解决方案：
- 设置 max_concurrent_requests: 1
- 增加 request_interval 到 10秒以上
- 检查图像分辨率设置
```

**3. 处理速度过慢**
```
解决方案：
- 适当增加 batch_size
- 减少 request_interval
- 检查网络延迟
```

### 监控命令

```bash
# 实时监控内存使用
watch -n 1 'free -h && echo "---" && ps aux | grep python | head -5'

# 监控GPU使用（如果有NVIDIA GPU）
watch -n 1 'nvidia-smi'

# 监控处理进度
tail -f logs/optimization.log | grep -E "(批次|内存|成功)"
```

## 📈 性能监控

### 关键指标

1. **内存使用率** - 应保持在75%以下
2. **处理速度** - 查询/分钟
3. **成功率** - 成功处理的查询比例
4. **缓存命中率** - 图像预处理缓存效率

### 日志分析

```bash
# 分析处理速度
grep "批次.*完成" logs/optimization.log | tail -10

# 分析内存使用
grep "内存使用" logs/optimization.log | tail -10

# 分析错误情况
grep "ERROR" logs/optimization.log | tail -10
```

## 🎯 最佳实践

1. **预处理测试** - 先用小数据集测试配置
2. **渐进式调优** - 从保守配置开始，逐步优化
3. **监控资源** - 实时关注内存和显存使用
4. **定期清理** - 处理完成后清理缓存文件
5. **备份配置** - 保存有效的配置文件

## 🔄 自动优化脚本

创建自动调优脚本 `auto_tune.py`：

```python
#!/usr/bin/env python3
import psutil
import yaml

def auto_tune_config():
    """根据系统配置自动调优"""
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    if memory_gb >= 32:
        batch_size = 5
        max_memory = 85.0
    elif memory_gb >= 16:
        batch_size = 3
        max_memory = 75.0
    else:
        batch_size = 2
        max_memory = 70.0
    
    config = {
        'optimization': {
            'batch_size': batch_size,
            'max_memory_usage': max_memory,
            'image_batch_size': min(3, batch_size)
        }
    }
    
    with open('auto_tuned_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"自动调优完成: 内存{memory_gb:.1f}GB -> batch_size={batch_size}")

if __name__ == "__main__":
    auto_tune_config()
```

使用自动调优：
```bash
python auto_tune.py
python batch_processor.py /input /output --config auto_tuned_config.yaml
```

