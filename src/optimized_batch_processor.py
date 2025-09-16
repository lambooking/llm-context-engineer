"""
优化的异步批量处理器
基于AsyncBatchProcessor，增加了内存优化和分批处理功能
"""

import asyncio
import json
import time
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from datetime import datetime
import yaml
import math

from loguru import logger
from asyncio_throttle import Throttler

from .async_batch_processor import AsyncBatchProcessor, BatchConfig, ConversationRecord
from .main_processor import LLMContextProcessor
from .image_preprocessor import EnhancedImagePreprocessor, EnhancedPreprocessingPipeline, create_enhanced_preprocessor_from_config, TifProcessResult
from .input_parser import InputParser
from .custom_vlm_client import EnhancedVLMClientFactory


@dataclass
class OptimizedBatchConfig(BatchConfig):
    """优化的批量处理配置"""
    batch_size: int = 10  # 每批处理的查询数量
    memory_threshold_mb: int = 1024  # 内存阈值（MB）
    enable_gc: bool = True  # 是否启用垃圾回收
    gc_interval: int = 5  # 垃圾回收间隔（批次）
    progressive_loading: bool = True  # 渐进式加载


class OptimizedBatchProcessor(AsyncBatchProcessor):
    """
    优化的异步批量处理器
    
    相比标准AsyncBatchProcessor的优化：
    1. 分批处理 - 避免同时处理过多查询导致内存溢出
    2. 内存管理 - 定期清理内存和垃圾回收
    3. 渐进式图像预处理 - 按需处理图像而不是一次性全部处理
    4. 更好的错误恢复 - 单个批次失败不会影响其他批次
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化优化的异步批量处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 调用父类初始化
        super().__init__(config_path)
        
        # 创建优化配置
        self.optimized_config = self._create_optimized_batch_config()
        
        # 批次计数器
        self.batch_counter = 0
        
        logger.info("优化版异步批量处理器初始化完成")
        logger.info(f"批次大小: {self.optimized_config.batch_size}")
        logger.info(f"内存阈值: {self.optimized_config.memory_threshold_mb}MB")
    
    def _create_optimized_batch_config(self) -> OptimizedBatchConfig:
        """创建优化的批量处理配置"""
        batch_settings = self.config.get('batch_processing', {})
        return OptimizedBatchConfig(
            # 继承基础配置
            max_concurrent_requests=batch_settings.get('max_concurrent_requests', 5),
            request_interval=batch_settings.get('request_interval', 0.1),
            retry_attempts=batch_settings.get('retry_attempts', 3),
            timeout=batch_settings.get('timeout', 60),
            save_conversations=batch_settings.get('save_conversations', True),
            streaming_output=batch_settings.get('streaming_output', True),
            output_dir=batch_settings.get('output_dir', 'output'),
            # 优化配置
            batch_size=batch_settings.get('batch_size', 10),
            memory_threshold_mb=batch_settings.get('memory_threshold_mb', 1024),
            enable_gc=batch_settings.get('enable_gc', True),
            gc_interval=batch_settings.get('gc_interval', 5),
            progressive_loading=batch_settings.get('progressive_loading', True)
        )
    
    def _split_queries_into_batches(self, queries: List[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
        """将查询分割成批次"""
        batch_size = self.optimized_config.batch_size
        for i in range(0, len(queries), batch_size):
            yield queries[i:i + batch_size]
    
    async def process_input_directory(self, input_path: str) -> Dict[str, Any]:
        """
        优化版的输入目录处理
        
        Args:
            input_path: 输入目录路径
            
        Returns:
            Dict: 处理结果摘要
        """
        start_time = time.time()
        
        # 解析输入
        logger.info("解析输入目录...")
        parsed_inputs = self.input_parser.parse_input_directory(input_path)
        queries = self.input_parser.convert_to_queries(parsed_inputs)
        
        if not queries:
            logger.error("没有找到有效的输入数据")
            return {'success': False, 'error': 'No valid input data found'}
        
        logger.info(f"解析到 {len(queries)} 个查询")
        
        # 分批处理
        batches = list(self._split_queries_into_batches(queries))
        logger.info(f"分为 {len(batches)} 个批次处理（每批 {self.optimized_config.batch_size} 个查询）")
        
        # 初始化结果收集器
        all_results = []
        total_success = 0
        total_failed = 0
        
        # 逐批处理
        for batch_idx, batch_queries in enumerate(batches):
            logger.info(f"处理第 {batch_idx + 1}/{len(batches)} 批次 ({len(batch_queries)} 个查询)")
            
            try:
                # 预处理当前批次的图像
                if self.optimized_config.progressive_loading:
                    await self._preprocess_batch_images(batch_queries)
                
                # 处理当前批次（支持流式输出）
                batch_results = await self._process_batch_optimized(batch_queries, batch_idx)
                all_results.extend(batch_results)
                
                # 统计成功失败
                batch_success = sum(1 for r in batch_results if r['success'])
                batch_failed = len(batch_results) - batch_success
                total_success += batch_success
                total_failed += batch_failed
                
                logger.info(f"批次 {batch_idx + 1} 完成: {batch_success}/{len(batch_results)} 成功")
                
                # 内存管理
                await self._manage_memory(batch_idx)
                
            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 处理失败: {e}")
                # 为失败的批次创建错误结果
                for i, query in enumerate(batch_queries):
                    all_results.append({
                        'success': False,
                        'query_index': batch_idx * self.optimized_config.batch_size + i,
                        'question': query.get('question', ''),
                        'error': f"批次处理失败: {str(e)}",
                        'processing_time': 0.0,
                        'metadata': query.get('metadata', {})
                    })
                total_failed += len(batch_queries)
        
        # 计算总体统计信息
        processing_time = time.time() - start_time
        
        summary = {
            'success': True,
            'input_path': input_path,
            'total_queries': len(queries),
            'successful': total_success,
            'failed': total_failed,
            'processing_time': processing_time,
            'batches_processed': len(batches),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存结果
        await self._save_results(summary, all_results)
        
        logger.info(f"优化批量处理完成: {total_success}/{len(queries)} 成功，用时 {processing_time:.2f}秒")
        return summary
    
    async def _preprocess_batch_images(self, batch_queries: List[Dict[str, Any]]):
        """预处理当前批次的图像"""
        batch_images = []
        for query in batch_queries:
            batch_images.extend(query['images'])
        
        if batch_images:
            logger.debug(f"预处理批次图像: {len(batch_images)} 张")
            self.preprocessing_pipeline.add_to_pipeline(batch_images)
    
    async def _process_batch_optimized(self, batch_queries: List[Dict[str, Any]], batch_idx: int, 
                                     progress_callback=None) -> List[Dict[str, Any]]:
        """优化的批次处理，支持流式输出"""
        # 创建限流器（每个批次独立）
        throttler = Throttler(
            rate_limit=self.optimized_config.max_concurrent_requests,
            period=1.0
        )
        
        # 创建当前批次的任务
        tasks = []
        for i, query in enumerate(batch_queries):
            query_index = batch_idx * self.optimized_config.batch_size + i
            # 正确创建任务对象而不是直接使用协程
            task = asyncio.create_task(self._process_single_query_with_throttle(throttler, query, query_index))
            tasks.append(task)
        
        # 改用更可靠的方式：直接使用gather处理所有任务
        processed_results = [None] * len(tasks)
        
        try:
            logger.debug(f"开始处理 {len(tasks)} 个任务...")
            
            # 使用gather等待所有任务，并处理异常
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果并实时显示
            for i, result in enumerate(results):
                query_index = batch_idx * self.optimized_config.batch_size + i
                
                if isinstance(result, Exception):
                    # 处理异常结果
                    error_result = {
                        'success': False,
                        'error': str(result),
                        'query_index': query_index,
                        'question': batch_queries[i].get('question', ''),
                        'processing_time': 0.0,
                        'metadata': batch_queries[i].get('metadata', {})
                    }
                    processed_results[i] = error_result
                    logger.error(f"查询 {query_index} 处理异常: {result}")
                    
                    # 显示错误结果
                    if progress_callback:
                        await progress_callback(error_result, query_index, len(tasks))
                    else:
                        self._display_single_result(error_result, query_index)
                else:
                    # 处理成功结果
                    processed_results[i] = result
                    
                    # 显示成功结果
                    if progress_callback:
                        await progress_callback(result, query_index, len(tasks))
                    else:
                        self._display_single_result(result, query_index)
                        
        except Exception as e:
            logger.error(f"批次处理完全失败: {e}")
            # 创建所有错误结果
            for i in range(len(tasks)):
                query_index = batch_idx * self.optimized_config.batch_size + i
                error_result = {
                    'success': False,
                    'error': f"批次处理失败: {str(e)}",
                    'query_index': query_index,
                    'question': batch_queries[i].get('question', ''),
                    'processing_time': 0.0,
                    'metadata': batch_queries[i].get('metadata', {})
                }
                processed_results[i] = error_result
        
        return processed_results
    
    def _display_single_result(self, result: Dict[str, Any], query_index: int):
        """显示单个查询结果（流式输出）"""
        print(f"\n{'='*60}")
        print(f"查询 #{query_index + 1} 完成")
        print(f"{'='*60}")
        
        if result['success']:
            print(f"问题: {result.get('question', 'N/A')}")
            print(f"问题类型: {result.get('question_type', 'N/A')}")
            print(f"图像数量: {result.get('images_count', 0)}")
            print(f"处理时间: {result.get('processing_time', 0):.2f}秒")
            
            if 'classification_confidence' in result:
                print(f"分类置信度: {result['classification_confidence']:.2f}")
            
            print(f"\n回答:")
            print(f"{result.get('answer', 'N/A')}")
            
            if 'usage' in result:
                usage = result['usage']
                print(f"\nToken使用:")
                print(f"  输入: {usage.get('prompt_tokens', 0)}")
                print(f"  输出: {usage.get('completion_tokens', 0)}")
                print(f"  总计: {usage.get('total_tokens', 0)}")
        else:
            print(f"❌ 处理失败")
            print(f"问题: {result.get('question', 'N/A')}")
            print(f"错误: {result.get('error', 'Unknown error')}")
        
        print(f"{'='*60}")
    
    async def _manage_memory(self, batch_idx: int):
        """内存管理"""
        self.batch_counter = batch_idx + 1
        
        # 定期垃圾回收
        if (self.optimized_config.enable_gc and 
            self.batch_counter % self.optimized_config.gc_interval == 0):
            
            logger.debug(f"执行垃圾回收 (批次 {self.batch_counter})")
            gc.collect()
        
        # 清理预处理缓存（可选）
        if hasattr(self.preprocessing_pipeline, 'clear_old_cache'):
            self.preprocessing_pipeline.clear_old_cache()
        
        # 简单的内存监控（如果可用）
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.optimized_config.memory_threshold_mb:
                logger.warning(f"内存使用量较高: {memory_mb:.1f}MB")
                # 强制垃圾回收
                gc.collect()
                
        except ImportError:
            # psutil不可用，跳过内存监控
            pass
    
    async def _call_vlm_with_retry(
        self, 
        question: str, 
        images: List[str], 
        force_type: Optional[Any] = None
    ) -> Dict[str, Any]:
        """带重试和超时的VLM API调用（优化版本）"""
        last_error = None
        timeout = self.batch_config.timeout
        
        for attempt in range(self.batch_config.retry_attempts):
            try:
                # 使用asyncio.wait_for添加超时控制
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        self.processor.process_query,
                        question,
                        images,
                        force_type
                    ),
                    timeout=timeout
                )
                
                if result['success']:
                    return result
                else:
                    last_error = result.get('error', 'Unknown error')
                    logger.warning(f"VLM API调用失败 (尝试 {attempt + 1}/{self.batch_config.retry_attempts}): {last_error}")
                    
            except asyncio.TimeoutError:
                last_error = f"请求超时 ({timeout}秒)"
                logger.warning(f"VLM API调用超时 (尝试 {attempt + 1}/{self.batch_config.retry_attempts}): {timeout}秒")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"VLM API调用失败 (尝试 {attempt + 1}/{self.batch_config.retry_attempts}): {e}")
                
            # 重试前等待，使用指数退避和抖动
            if attempt < self.batch_config.retry_attempts - 1:
                base_delay = min(2 ** attempt, 30)  # 最大30秒
                jitter = base_delay * 0.1  # 10%抖动
                import random
                delay = base_delay + (jitter * (2 * random.random() - 1))
                logger.info(f"等待 {delay:.1f}秒 后重试...")
                await asyncio.sleep(delay)
        
        return {
            'success': False,
            'error': f"所有重试失败: {last_error}"
        }

    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'batch_size': self.optimized_config.batch_size,
            'batches_processed': self.batch_counter,
            'memory_threshold_mb': self.optimized_config.memory_threshold_mb,
            'gc_enabled': self.optimized_config.enable_gc,
            'progressive_loading': self.optimized_config.progressive_loading
        }


def create_optimized_processor_from_config(config_path: str = "config/config.yaml") -> OptimizedBatchProcessor:
    """从配置文件创建优化的异步批量处理器"""
    return OptimizedBatchProcessor(config_path)
