#!/usr/bin/env python3
"""
LLM Context Engineer - 主入口文件
基于VLM服务的智能问答系统
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse
from loguru import logger

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main_processor import ProcessorFactory
from src.constants import QuestionType


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """设置日志配置"""
    logger.remove()
    
    # 控制台输出
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 文件输出
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM Context Engineer - 基于VLM服务的智能问答系统")
    parser.add_argument("question", help="用户问题")
    parser.add_argument("-i", "--images", nargs="*", help="图像文件路径列表")
    parser.add_argument("-t", "--type", choices=[t.name for t in QuestionType], help="强制指定问题类型")
    parser.add_argument("--preview", action="store_true", help="仅预览分类结果，不调用API")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--log-file", help="日志文件路径")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    
    try:
        # 创建处理器
        processor = ProcessorFactory.create_from_env()
        logger.info("处理器初始化完成")
        
        # 处理图像路径
        images = []
        if args.images:
            for img_path in args.images:
                if os.path.exists(img_path):
                    images.append(img_path)
                    logger.info(f"添加图像: {img_path}")
                else:
                    logger.warning(f"图像文件不存在: {img_path}")
        
        # 预览模式
        if args.preview:
            result = processor.get_classification_preview(args.question, len(images))
            print("\n=== 分类预览结果 ===")
            print(f"问题: {result['question']}")
            print(f"预测类型: {result['predicted_type']}")
            print(f"置信度: {result['confidence']:.2f}")
            print(f"图像数量: {result['image_count']}")
            print("\n所有类型置信度:")
            for qtype, confidence in result['all_confidences'].items():
                print(f"  {qtype}: {confidence:.2f}")
            return
        
        # 强制类型转换
        force_type = None
        if args.type:
            force_type = QuestionType[args.type]
            logger.info(f"使用强制指定类型: {force_type.value}")
        
        # 处理查询
        logger.info("开始处理查询...")
        result = processor.process_query(
            question=args.question,
            images=images,
            force_type=force_type
        )
        
        # 输出结果
        if result['success']:
            print("\n=== 处理结果 ===")
            print(f"问题: {result['question']}")
            print(f"问题类型: {result['question_type']}")
            if 'classification_confidence' in result:
                print(f"分类置信度: {result['classification_confidence']:.2f}")
            print(f"图像数量: {result['images_count']}")
            print(f"\n回答:\n{result['answer']}")
            print(f"\n使用情况:")
            usage = result['usage']
            print(f"  输入tokens: {usage['prompt_tokens']}")
            print(f"  输出tokens: {usage['completion_tokens']}")
            print(f"  总tokens: {usage['total_tokens']}")
        else:
            print(f"\n处理失败: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
