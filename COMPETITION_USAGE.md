# LLM Context Engineer - 比赛使用说明

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API密钥
cp env.example .env
# 编辑.env文件，设置你的VLM API密钥
```

### 2. 输入数据格式

输入数据必须按照以下目录结构组织：

```
input_data/
├── QA/
│   ├── image/
│   │   ├── 0.tif
│   │   ├── 1.tif
│   │   └── ...
│   └── question/
│       ├── 0.txt
│       ├── 1.txt
│       └── ...
├── Image_caption/
│   ├── image/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── question/
│       ├── 0.txt
│       ├── 1.txt
│       └── ...
└── Change_caption/
    ├── image1/
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    ├── image2/
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    └── question/
        ├── 0.txt
        ├── 1.txt
        └── ...
```

### 3. 运行批量处理

```bash
# 基本用法
python batch_processor.py /path/to/input_data /path/to/output_path

# 预览模式（不调用API）
python batch_processor.py /path/to/input_data /path/to/output_path --dry-run

# 仅验证输入结构
python batch_processor.py /path/to/input_data /path/to/output_path --validate

# 指定配置文件
python batch_processor.py /path/to/input_data /path/to/output_path --config config/config.yaml

# 调试模式
python batch_processor.py /path/to/input_data /path/to/output_path --log-level DEBUG
```

### 4. 输出结果

程序会在指定的输出路径下创建以下结构：

```
output_path/
├── QA/
│   ├── 0.txt      # 基础问答和图表分析结果
│   ├── 1.txt
│   └── ...
├── Image_caption/
│   ├── 0.txt      # 图像描述结果
│   ├── 1.txt
│   └── ...
├── Change_caption/
│   ├── 0.txt      # 对比分析结果
│   ├── 1.txt
│   └── ...
├── batch_results_20240910_143022.json    # 详细处理结果
├── conversations_20240910_143022.json    # 对话记录
└── cache_info_20240910_143022.json       # 缓存信息
```

**重要说明**：
- 所有txt文件均为UTF-8编码
- 文件编号从0开始，按处理顺序递增
- QA文件夹包含基础问答和图表分析的结果
- 失败的查询不会生成txt文件

### 5. 配置说明

主要配置项（`config/config.yaml`）：

```yaml
# VLM服务配置
vlm:
  provider: "openai"
  api_key: "${VLM_API_KEY}"
  model: "gpt-4-vision-preview"
  max_tokens: 4000

# 批量处理配置
batch_processing:
  max_concurrent_requests: 5    # 并发数量
  request_interval: 0.1         # 请求间隔
  retry_attempts: 3             # 重试次数

# 图像预处理配置
image_preprocessing:
  enabled: true
  max_resolution: [2048, 2048]  # 最大分辨率
  quality: 85                   # JPEG质量
```

### 6. 常见问题

**Q: 如何处理大分辨率遥感图像？**
A: 系统自动启用图像预处理，会将大图像压缩到合适的分辨率，同时保持质量。

**Q: 如何控制并发数量？**
A: 在配置文件中修改`batch_processing.max_concurrent_requests`参数。

**Q: 处理失败怎么办？**
A: 系统会自动重试，失败的查询会记录在日志中，可以检查具体错误信息。

**Q: 如何查看详细的处理日志？**
A: 使用`--log-level DEBUG --log-file logs/debug.log`参数。

### 7. 性能优化建议

1. **并发设置**: 根据API服务的限制调整并发数量
2. **图像预处理**: 启用预处理可以减少API调用时间
3. **缓存利用**: 相同的图像会自动使用缓存，避免重复处理
4. **批量处理**: 一次处理多个任务比单独处理更高效

### 8. 故障排除

```bash
# 检查输入格式
python batch_processor.py /path/to/input --validate

# 预览处理内容
python batch_processor.py /path/to/input /path/to/output --dry-run

# 查看详细日志
python batch_processor.py /path/to/input /path/to/output --log-level DEBUG
```
