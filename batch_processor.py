#!/usr/bin/env python3
"""
批量处理器 - 处理结构化输入格式的数据
"""

import os
import sys
import argparse
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.input_parser import InputParser
from src.main_processor import ProcessorFactory
from src.async_batch_processor import AsyncBatchProcessor
from src.optimized_batch_processor import OptimizedBatchProcessor
from src.constants import QuestionType


def setup_logging(level: str = "INFO", log_file: str = None):
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


def validate_input(input_path: str) -> bool:
    """验证输入目录结构"""
    parser = InputParser()
    validation_result = parser.validate_input_structure(input_path)
    
    print("=== 输入结构验证 ===")
    print(f"验证结果: {'通过' if validation_result['valid'] else '失败'}")
    
    if validation_result['errors']:
        print("\n错误:")
        for error in validation_result['errors']:
            print(f"  ❌ {error}")
    
    if validation_result['warnings']:
        print("\n警告:")
        for warning in validation_result['warnings']:
            print(f"  ⚠️  {warning}")
    
    if validation_result['summary']:
        print("\n文件统计:")
        for type_name, counts in validation_result['summary'].items():
            print(f"  {type_name}:")
            for subdir, count in counts.items():
                print(f"    {subdir}: {count} 文件")
    
    return validation_result['valid']


async def process_batch_async(input_path: str, output_path: str = None, config_path: str = "config/config.yaml", dry_run: bool = False, optimized: bool = True):
    """异步批量处理输入数据"""
    if dry_run:
        # 预览模式使用同步解析
        parser = InputParser()
        parsed_inputs = parser.parse_input_directory(input_path)
        queries = parser.convert_to_queries(parsed_inputs)
        
        print("\n=== 预览模式 ===")
        print(f"输入路径: {input_path}")
        print(f"输出路径: {output_path or 'output'}")
        for i, query in enumerate(queries[:5]):  # 只显示前5个
            print(f"\n查询 {i+1}:")
            print(f"  类型: {query['force_type'].value}")
            print(f"  问题: {query['question'][:100]}...")
            print(f"  图像数量: {len(query['images'])}")
            print(f"  来源: {query['metadata']['source_dir']}")
        
        if len(queries) > 5:
            print(f"\n... 还有 {len(queries) - 5} 个查询")
        return
    
    # 使用异步批量处理器
    try:
        # 准备配置
        final_config_path = config_path
        temp_config_path = None
        
        # 如果指定了输出路径，创建自定义配置
        if output_path:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 修改输出路径
            if 'batch_processing' not in config:
                config['batch_processing'] = {}
            config['batch_processing']['output_dir'] = output_path
            
            # 创建临时配置文件
            temp_config_path = "temp_config.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
            
            final_config_path = temp_config_path
        
        # 选择处理器类型
        if optimized:
            processor = OptimizedBatchProcessor(final_config_path)
            logger.info("使用优化版批量处理器")
        else:
            processor = AsyncBatchProcessor(final_config_path)
            logger.info("使用标准版批量处理器")
        
        logger.info("异步批量处理器初始化完成")
        logger.info(f"输入路径: {input_path}")
        logger.info(f"输出路径: {output_path or processor.output_dir}")
        
        # 处理输入目录
        summary = await processor.process_input_directory(input_path)
        
        # 在处理完成后清理临时配置文件
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        if summary['success']:
            print(f"\n=== 异步批量处理完成 ===")
            print(f"输入路径: {input_path}")
            print(f"输出路径: {output_path or processor.output_dir}")
            print(f"总查询数: {summary['total_queries']}")
            print(f"成功: {summary['successful']}")
            print(f"失败: {summary['failed']}")
            print(f"成功率: {summary['successful']/summary['total_queries']*100:.1f}%")
            print(f"处理时间: {summary['processing_time']:.2f}秒")
            print(f"比赛结果已按类型保存到子文件夹: QA/, Image_caption/, Change_caption/")
        else:
            print(f"批量处理失败: {summary.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"异步批量处理失败: {e}")
        return


def process_batch(input_path: str, output_path: str, dry_run: bool = False, optimized: bool = True):
    """批量处理输入数据（兼容性包装）"""
    # 运行异步处理
    asyncio.run(process_batch_async(input_path, output_path, "config/config.yaml", dry_run, optimized))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LLM Context Engineer - 批量处理器（比赛版本）",
        epilog="""
使用示例:
  python batch_processor.py /path/to/input_data /path/to/output_path
  python batch_processor.py /path/to/input_data /path/to/output_path --dry-run
  python batch_processor.py /path/to/input_data /path/to/output_path --validate
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_path", help="输入数据目录路径（必须包含QA/、Image_caption/、Change_caption/等子目录）")
    parser.add_argument("output_path", help="输出目录路径（比赛要求的/output_path路径）")
    parser.add_argument("--validate", action="store_true", help="仅验证输入结构")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际调用API")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--log-file", help="日志文件路径")
    parser.add_argument("--no-optimization", action="store_true", help="禁用优化模式，使用标准处理器")
    parser.add_argument("--batch-size", type=int, help="批次大小（覆盖配置文件设置）")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    
    print("=== LLM Context Engineer 批量处理器 ===")
    print(f"输入路径: {args.input_path}")
    print(f"输出路径: {args.output_path}")
    print(f"配置文件: {args.config}")
    
    try:
        # 验证输入
        if not validate_input(args.input_path):
            logger.error("输入验证失败")
            sys.exit(1)
        
        if args.validate:
            print("✅ 输入验证通过")
            return
        
        # 批量处理
        optimized = not args.no_optimization
        if optimized:
            print("🚀 使用优化模式 - 内存友好的分批处理")
        else:
            print("⚡ 使用标准模式 - 传统并发处理")
            
        process_batch(args.input_path, args.output_path, args.dry_run, optimized)
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
