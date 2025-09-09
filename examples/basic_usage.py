#!/usr/bin/env python3
"""
基础使用示例
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.main_processor import ProcessorFactory
from src.constants import QuestionType


def basic_example():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 从环境变量创建处理器
    processor = ProcessorFactory.create_from_env()
    
    # 示例1: 基础问答
    print("\n1. 基础问答示例:")
    question = "这张图片中有什么？"
    images = ["path/to/your/image.jpg"]  # 替换为实际图片路径
    
    result = processor.process_query(question, images)
    if result['success']:
        print(f"问题类型: {result['question_type']}")
        print(f"回答: {result['answer']}")
    else:
        print(f"处理失败: {result['error']}")


def classification_example():
    """分类示例"""
    print("\n=== 分类示例 ===")
    
    processor = ProcessorFactory.create_from_env()
    
    # 不同类型的问题示例
    questions = [
        ("描述这张图片的内容", 1),
        ("比较这两张图片的差异", 2),
        ("分析这个图表中的数据趋势", 1),
        ("这个公式是什么意思？", 1)
    ]
    
    for question, image_count in questions:
        preview = processor.get_classification_preview(question, image_count)
        print(f"\n问题: {question}")
        print(f"预测类型: {preview['predicted_type']}")
        print(f"置信度: {preview['confidence']:.2f}")


def batch_processing_example():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    processor = ProcessorFactory.create_from_env()
    
    # 批量查询
    queries = [
        {
            "question": "描述这张图片",
            "images": ["image1.jpg"],
            "force_type": QuestionType.IMAGE_DESC
        },
        {
            "question": "分析这个图表",
            "images": ["chart.png"],
            "force_type": QuestionType.CHART_ANALYSIS
        }
    ]
    
    results = processor.batch_process(queries)
    
    for i, result in enumerate(results):
        print(f"\n查询 {i+1}:")
        if result['success']:
            print(f"  类型: {result['question_type']}")
            print(f"  回答: {result['answer'][:100]}...")
        else:
            print(f"  错误: {result['error']}")


def prompt_management_example():
    """Prompt管理示例"""
    print("\n=== Prompt管理示例 ===")
    
    processor = ProcessorFactory.create_from_env()
    
    # 查看当前的Prompt
    current_prompt = processor.prompt_manager.get_prompt(
        QuestionType.IMAGE_DESC,
        question="示例问题"
    )
    print(f"当前IMAGE_DESC的Prompt:\n{current_prompt[:200]}...")
    
    # 更新Prompt
    new_prompt = """请详细描述这张图像，包括：
1. 主要对象和场景
2. 颜色和构图
3. 情感氛围
4. 技术细节

用户问题：{question}"""
    
    processor.update_prompt(QuestionType.IMAGE_DESC, new_prompt)
    print("\n已更新IMAGE_DESC的Prompt")
    
    # 重新加载Prompt
    processor.reload_prompts()
    print("已重新加载所有Prompt")


def custom_client_example():
    """自定义客户端示例"""
    print("\n=== 自定义客户端示例 ===")
    
    # 自定义配置
    config = {
        'vlm': {
            'provider': 'openai',
            'api_key': 'your_api_key',
            'base_url': 'https://api.openai.com/v1',
            'model': 'gpt-4-vision-preview'
        },
        'prompts_dir': 'prompts'
    }
    
    processor = ProcessorFactory.create_processor(config)
    
    # 获取统计信息
    stats = processor.get_stats()
    print("处理器统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # 设置环境变量示例（实际使用时应该在.env文件中设置）
    os.environ.setdefault('VLM_API_KEY', 'your_api_key_here')
    os.environ.setdefault('VLM_PROVIDER', 'openai')
    
    try:
        basic_example()
        classification_example()
        # batch_processing_example()  # 需要实际图片文件
        prompt_management_example()
        custom_client_example()
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        print("请确保已正确设置环境变量和API密钥")
