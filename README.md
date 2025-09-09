# LLM Context Engineer

基于VLM服务的智能问答系统，支持多种问题类型的自动分类和专业化处理。

## 功能特性

-  **智能问题分类**: 自动识别四种问题类型（基础问答、图像描述、对比分析、图表分析）
-  **VLM API集成**: 支持GPT-4V格式的API调用，兼容多种服务提供商
-  **XML格式Prompt管理**: 灵活的Prompt模板系统，支持动态加载和修改
-  **专业化响应**: 针对不同问题类型提供优化的处理流程
-  **批量处理**: 支持多查询批量处理，提高效率
-  **Web API**: 提供REST API服务，方便集成

## 问题类型

| 类型 | 说明 | 示例 |
|------|------|------|
| 基础问答 | 一般性问题回答 | "这是什么？", "请解释一下" |
| 图像描述 | 图像内容描述 | "描述这张图片", "图像显示了什么" |
| 对比分析 | 多图像对比分析 | "比较这两张图片", "有什么不同" |
| 计数 | 图分析计数 | "分析这个图表", "计数" |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `env.example` 为 `.env` 并填入你的配置：

```bash
cp env.example .env
```

编辑 `.env` 文件：

```env
VLM_PROVIDER=openai
VLM_API_KEY=your_api_key_here
VLM_BASE_URL=https://api.openai.com/v1
VLM_MODEL=gpt-4-vision-preview
```

### 3. 基础使用

#### 命令行使用

```bash
# 基础问答
python main.py "这张图片中有什么？" -i image.jpg

# 强制指定类型
python main.py "分析数据" -i chart.png -t CHART_ANALYSIS

# 预览分类结果
python main.py "描述图像" --preview

# 多图像对比
python main.py "比较这些图片的差异" -i img1.jpg img2.jpg
```

#### Python代码使用

```python
from src.main_processor import ProcessorFactory
from src.constants import QuestionType

# 创建处理器
processor = ProcessorFactory.create_from_env()

# 处理查询
result = processor.process_query(
    question="描述这张图片的内容",
    images=["path/to/image.jpg"]
)

if result['success']:
    print(f"问题类型: {result['question_type']}")
    print(f"回答: {result['answer']}")
else:
    print(f"处理失败: {result['error']}")
```

### 4. Web API服务

启动API服务：

```bash
python examples/web_api.py
```

API端点：

- `GET /health` - 健康检查
- `POST /query` - 处理单个查询
- `POST /classify` - 问题分类预览
- `POST /batch` - 批量处理
- `GET /prompts` - 获取Prompt模板
- `PUT /prompts/<type>` - 更新Prompt模板

#### API使用示例

```bash
# 单个查询
curl -X POST http://localhost:5000/query \
  -F "question=描述这张图片" \
  -F "images=@image.jpg"

# 分类预览
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"question": "分析图表数据", "image_count": 1}'
```

## 项目结构

```
llm-context-engineer/
├── src/                          # 源代码
│   ├── constants.py              # 常量定义
│   ├── question_classifier.py    # 问题分类器
│   ├── prompt_manager.py         # Prompt管理器
│   ├── vlm_client.py            # VLM API客户端
│   └── main_processor.py        # 主处理器
├── prompts/                      # Prompt模板文件(XML格式)
├── config/                       # 配置文件
├── examples/                     # 使用示例
├── tests/                        # 测试文件
├── logs/                         # 日志文件
├── main.py                       # 命令行入口
├── requirements.txt              # 依赖列表
└── README.md                     # 项目文档
```

## Prompt管理

系统使用XML格式管理Prompt模板，支持动态加载和修改：

### XML格式示例

```xml
<?xml version='1.0' encoding='utf-8'?>
<prompt_template type="图像描述" name="IMAGE_DESC">
  <description>用于图像描述的Prompt模板</description>
  <prompt>请详细描述这张图像的内容。包括：
1. 主要物体和场景
2. 颜色、形状、位置等视觉特征
3. 任何可见的文字或标识
4. 整体的氛围和风格

请用清晰、有条理的语言进行描述。</prompt>
  <parameters>
    <parameter name="question">用户问题</parameter>
  </parameters>
</prompt_template>
```

### 程序化管理

```python
# 更新Prompt
processor.update_prompt(QuestionType.IMAGE_DESC, new_prompt_text)

# 重新加载所有Prompt
processor.reload_prompts()

# 获取当前Prompt
current_prompt = processor.prompt_manager.get_prompt(
    QuestionType.IMAGE_DESC, 
    question="用户问题"
)
```

## 高级功能

### 批量处理

```python
queries = [
    {
        "question": "描述图片",
        "images": ["img1.jpg"],
        "force_type": QuestionType.IMAGE_DESC
    },
    {
        "question": "分析数据",
        "images": ["chart.png"]
    }
]

results = processor.batch_process(queries)
```

### 自定义客户端

```python
config = {
    'vlm': {
        'provider': 'azure',
        'api_key': 'your_azure_key',
        'base_url': 'https://your-resource.openai.azure.com',
        'model': 'gpt-4-vision'
    }
}

processor = ProcessorFactory.create_processor(config)
```

### 分类置信度分析

```python
preview = processor.get_classification_preview(
    "分析这个图表的趋势", 
    image_count=1
)

print(f"预测类型: {preview['predicted_type']}")
print(f"置信度: {preview['confidence']}")
print("所有类型置信度:")
for qtype, confidence in preview['all_confidences'].items():
    print(f"  {qtype}: {confidence:.2f}")
```

## 测试

运行测试：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python tests/test_classifier.py
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| VLM_PROVIDER | VLM服务提供商 | openai |
| VLM_API_KEY | API密钥 | 必需 |
| VLM_BASE_URL | API基础URL | https://api.openai.com/v1 |
| VLM_MODEL | 使用的模型 | gpt-4-vision-preview |
| PROMPTS_DIR | Prompt目录 | prompts |
| LOG_LEVEL | 日志级别 | INFO |

### 配置文件

支持YAML格式的配置文件 `config/config.yaml`，详细配置选项请参考示例文件。

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 更新日志

### v1.0.0
- 初始版本发布
- 支持四种问题类型分类
- XML格式Prompt管理
- VLM API集成
- Web API服务
- 批量处理功能