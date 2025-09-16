"""
异步批量处理器
支持并发控制、图像预处理流水线和对话信息保存
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml

from loguru import logger
from asyncio_throttle import Throttler

from .main_processor import LLMContextProcessor
from .image_preprocessor import EnhancedImagePreprocessor, EnhancedPreprocessingPipeline, create_enhanced_preprocessor_from_config, TifProcessResult
from .input_parser import InputParser
from .custom_vlm_client import EnhancedVLMClientFactory


@dataclass
class BatchConfig:
    """批量处理配置"""
    max_concurrent_requests: int = 5
    request_interval: float = 0.1
    retry_attempts: int = 3
    timeout: int = 60
    save_conversations: bool = True
    streaming_output: bool = True  # 流式输出开关
    output_dir: str = "output"


@dataclass
class ConversationRecord:
    """对话记录"""
    timestamp: str
    question: str
    question_type: str
    images: List[str]
    processed_images: List[str]
    prompt_used: str
    response: str
    success: bool
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class AsyncBatchProcessor:
    """异步批量处理器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化异步批量处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.batch_config = self._create_batch_config()
        
        # 初始化组件
        self.input_parser = InputParser()
        self.image_preprocessor = create_enhanced_preprocessor_from_config(self.config)
        self.preprocessing_pipeline = EnhancedPreprocessingPipeline(self.image_preprocessor)
        
        # 创建VLM客户端
        vlm_config = self.config.get('vlm', {})
        provider = vlm_config.get('provider', 'openai')
        # 从配置中移除provider，避免重复传递
        vlm_config_clean = {k: v for k, v in vlm_config.items() if k != 'provider'}
        # 添加batch_processing中的timeout配置到VLM客户端配置
        vlm_config_clean['timeout'] = self.batch_config.timeout
        self.vlm_client = EnhancedVLMClientFactory.create_client(provider, **vlm_config_clean)
        
        # 创建主处理器
        self.processor = LLMContextProcessor(
            self.vlm_client, 
            self.config.get('prompts_dir', 'prompts')
        )
        
        # 输出目录
        self.output_dir = Path(self.batch_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 对话记录
        self.conversations: List[ConversationRecord] = []
        
        logger.info("异步批量处理器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {e}")
            return {}
    
    def _create_batch_config(self) -> BatchConfig:
        """创建批量处理配置"""
        batch_settings = self.config.get('batch_processing', {})
        return BatchConfig(
            max_concurrent_requests=batch_settings.get('max_concurrent_requests', 5),
            request_interval=batch_settings.get('request_interval', 0.1),
            retry_attempts=batch_settings.get('retry_attempts', 3),
            timeout=batch_settings.get('timeout', 60),
            save_conversations=batch_settings.get('save_conversations', True),
            streaming_output=batch_settings.get('streaming_output', True),
            output_dir=batch_settings.get('output_dir', 'output')
        )
    
    async def process_input_directory(self, input_path: str) -> Dict[str, Any]:
        """
        处理输入目录
        
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
        
        # 预处理图像（启动流水线）
        logger.info("启动图像预处理流水线...")
        all_images = []
        for query in queries:
            all_images.extend(query['images'])
        
        if all_images:
            self.preprocessing_pipeline.add_to_pipeline(all_images)
        
        # 异步处理查询
        logger.info("开始异步批量处理...")
        results = await self._process_queries_async(queries)
        
        # 计算统计信息
        processing_time = time.time() - start_time
        success_count = sum(1 for r in results if r['success'])
        
        # 保存结果
        summary = {
            'success': True,
            'input_path': input_path,
            'total_queries': len(queries),
            'successful': success_count,
            'failed': len(queries) - success_count,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存对话记录和结果
        await self._save_results(summary, results)
        
        logger.info(f"批量处理完成: {success_count}/{len(queries)} 成功")
        return summary
    
    async def _process_queries_async(self, queries: List[Dict[str, Any]], progress_callback=None) -> List[Dict[str, Any]]:
        """异步处理查询列表，支持流式输出"""
        # 创建限流器
        throttler = Throttler(
            rate_limit=self.batch_config.max_concurrent_requests,
            period=1.0
        )
        
        # 创建任务
        tasks = []
        for i, query in enumerate(queries):
            # 正确创建任务对象而不是直接使用协程
            task = asyncio.create_task(self._process_single_query_with_throttle(throttler, query, i))
            tasks.append(task)
        
        # 改用更可靠的方式：直接使用gather处理所有任务
        processed_results = [None] * len(tasks)
        
        try:
            logger.debug(f"开始处理 {len(tasks)} 个任务...")
            
            # 使用gather等待所有任务，并处理异常
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果并实时显示
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # 处理异常结果
                    error_result = {
                        'success': False,
                        'error': str(result),
                        'query_index': i,
                        'question': queries[i].get('question', ''),
                        'processing_time': 0.0,
                        'metadata': queries[i].get('metadata', {})
                    }
                    processed_results[i] = error_result
                    logger.error(f"查询 {i} 处理异常: {result}")
                    
                    # 显示错误结果
                    if progress_callback:
                        await progress_callback(error_result, i, len(tasks))
                    else:
                        self._display_single_result(error_result, i)
                else:
                    # 处理成功结果
                    processed_results[i] = result
                    
                    # 显示成功结果
                    if progress_callback:
                        await progress_callback(result, i, len(tasks))
                    else:
                        self._display_single_result(result, i)
                        
        except Exception as e:
            logger.error(f"批次处理完全失败: {e}")
            # 创建所有错误结果
            for i in range(len(tasks)):
                error_result = {
                    'success': False,
                    'error': f"批次处理失败: {str(e)}",
                    'query_index': i,
                    'question': queries[i].get('question', ''),
                    'processing_time': 0.0,
                    'metadata': queries[i].get('metadata', {})
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
    
    async def _process_single_query_with_throttle(
        self, 
        throttler: Throttler, 
        query: Dict[str, Any], 
        index: int
    ) -> Dict[str, Any]:
        """带限流的单个查询处理"""
        async with throttler:
            # 添加请求间隔
            if self.batch_config.request_interval > 0:
                await asyncio.sleep(self.batch_config.request_interval)
            
            return await self._process_single_query_async(query, index)
    
    async def _process_single_query_async(self, query: Dict[str, Any], index: int) -> Dict[str, Any]:
        """异步处理单个查询（支持多问题）"""
        start_time = time.time()
        
        # 检查是否为多问题查询
        if query.get('is_multi_question', False):
            return await self._process_multi_question_query_async(query, index)
        
        # 单问题处理逻辑
        question = query['question']
        original_images = query['images']
        force_type = query.get('force_type')
        metadata = query.get('metadata', {})
        
        logger.debug(f"处理查询 {index}: {question[:50]}...")
        
        try:
            # 等待图像预处理完成
            processed_images = []
            tif_results = []  # 存储TIF处理结果
            
            for img_path in original_images:
                processed_result = self.preprocessing_pipeline.get_processed_result(
                    img_path, 
                    timeout=600.0
                )
                
                if isinstance(processed_result, TifProcessResult):
                    # TIF图像：使用resize后的整图路径
                    processed_images.append(processed_result.resized_image_path)
                    tif_results.append(processed_result)
                    logger.debug(f"TIF图像处理完成: {img_path} -> {len(processed_result.tiles)} 个切片")
                else:
                    # 常规图像：直接使用路径
                    processed_images.append(str(processed_result))
                    tif_results.append(None)
            
            # 获取使用的Prompt
            if force_type:
                question_type = force_type
            else:
                question_type = self.processor.classifier.classify(question, len(original_images))
            
            prompt_used = self.processor.prompt_manager.get_prompt(question_type, question=question)
            
            # 调用VLM API（带重试）
            result = await self._call_vlm_with_retry(question, processed_images, force_type)
            
            processing_time = time.time() - start_time
            
            # 准备扩展的元数据
            extended_metadata = {**metadata}
            
            # 添加TIF处理信息
            tif_info = []
            for i, tif_result in enumerate(tif_results):
                if tif_result:
                    tif_info.append({
                        'original_path': original_images[i],
                        'tiles_count': len(tif_result.tiles),
                        'processing_time': tif_result.processing_time,
                        'json_path': tif_result.json_path
                    })
            
            if tif_info:
                extended_metadata['tif_processing'] = tif_info
            
            # 记录对话
            conversation = ConversationRecord(
                timestamp=datetime.now().isoformat(),
                question=question,
                question_type=question_type.value,
                images=original_images,
                processed_images=processed_images,
                prompt_used=prompt_used,
                response=result.get('answer', '') if result['success'] else '',
                success=result['success'],
                error=result.get('error'),
                usage=result.get('usage'),
                processing_time=processing_time,
                metadata=extended_metadata
            )
            
            self.conversations.append(conversation)
            
            # 构建返回结果
            return {
                'success': result['success'],
                'query_index': index,
                'question': question,
                'question_type': question_type.value,
                'answer': result.get('answer', ''),
                'images_count': len(original_images),
                'processed_images_count': len(processed_images),
                'processing_time': processing_time,
                'usage': result.get('usage', {}),
                'error': result.get('error'),
                'metadata': metadata
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"查询 {index} 处理失败: {e}")
            
            # 记录失败的对话
            conversation = ConversationRecord(
                timestamp=datetime.now().isoformat(),
                question=question,
                question_type=force_type.value if force_type else 'unknown',
                images=original_images,
                processed_images=[],
                prompt_used='',
                response='',
                success=False,
                error=str(e),
                processing_time=processing_time,
                metadata=metadata
            )
            
            self.conversations.append(conversation)
            
            return {
                'success': False,
                'query_index': index,
                'question': question,
                'error': str(e),
                'processing_time': processing_time,
                'metadata': metadata
            }
    
    async def _call_vlm_with_retry(
        self, 
        question: str, 
        images: List[str], 
        force_type: Optional[Any] = None
    ) -> Dict[str, Any]:
        """带重试的VLM API调用"""
        last_error = None
        
        for attempt in range(self.batch_config.retry_attempts):
            try:
                # 这里需要将同步调用转换为异步
                # 在实际实现中，可能需要修改VLMClient支持异步
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.processor.process_query,
                    question,
                    images,
                    force_type
                )
                
                if result['success']:
                    return result
                else:
                    last_error = result.get('error', 'Unknown error')
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"VLM API调用失败 (尝试 {attempt + 1}/{self.batch_config.retry_attempts}): {e}")
                
                if attempt < self.batch_config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return {
            'success': False,
            'error': f"所有重试失败: {last_error}"
        }
    
    async def _save_results(self, summary: Dict[str, Any], results: List[Dict[str, Any]]):
        """保存处理结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存主要结果
        results_file = self.output_dir / f"batch_results_{timestamp}.json"
        output_data = {
            'summary': summary,
            'results': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量处理结果已保存: {results_file}")
        
        # 按任务类型保存txt结果文件（比赛要求）
        await self._save_competition_results(results)
        
        # 保存对话记录
        if self.batch_config.save_conversations:
            conversations_file = self.output_dir / f"conversations_{timestamp}.json"
            conversations_data = {
                'metadata': {
                    'total_conversations': len(self.conversations),
                    'timestamp': datetime.now().isoformat(),
                    'processing_summary': summary
                },
                'conversations': [asdict(conv) for conv in self.conversations]
            }
            
            with open(conversations_file, 'w', encoding='utf-8') as f:
                json.dump(conversations_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"对话记录已保存: {conversations_file}")
        
        # 保存图像预处理缓存信息
        cache_info = self.image_preprocessor.get_cache_info()
        cache_info_file = self.output_dir / f"cache_info_{timestamp}.json"
        
        with open(cache_info_file, 'w', encoding='utf-8') as f:
            json.dump(cache_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"缓存信息已保存: {cache_info_file}")
    
    async def _save_competition_results(self, results: List[Dict[str, Any]]):
        """
        按比赛要求保存结果文件
        根据任务类型分别保存到QA、Image_caption、Change_caption子文件夹
        """
        # 任务类型到文件夹名称的映射
        type_to_folder = {
            '基础问答': 'QA',
            '图像描述': 'Image_caption',
            '对照对比问答类/素异描述': 'Change_caption',
            '图表分析/计算': 'QA'  # 图表分析归类到QA
        }
        
        # 创建子文件夹
        for folder_name in ['QA', 'Image_caption', 'Change_caption']:
            folder_path = self.output_dir / folder_name
            folder_path.mkdir(exist_ok=True)
        
        # 按类型统计和保存
        type_counters = {'QA': 0, 'Image_caption': 0, 'Change_caption': 0}
        
        for result in results:
            if not result['success']:
                continue
                
            question_type = result.get('question_type', '基础问答')
            folder_name = type_to_folder.get(question_type, 'QA')
            
            # 获取文件编号
            file_number = type_counters[folder_name]
            type_counters[folder_name] += 1
            
            # 构建输出文件路径
            output_file = self.output_dir / folder_name / f"{file_number}.txt"
            
            # 准备输出内容 - 使用text_truth格式
            answer = result.get('answer', '').strip()
            if not answer:
                answer = "处理失败，无法生成回答。"
            
            # 使用text_truth格式
            output_content = f"text_truth: {answer}"
            
            # 保存为UTF-8编码的txt文件
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output_content)
                
                logger.debug(f"保存比赛结果: {output_file}")
                
            except Exception as e:
                logger.error(f"保存比赛结果失败 {output_file}: {e}")
        
        # 记录保存统计
        logger.info("比赛结果文件保存完成:")
        for folder_name, count in type_counters.items():
            logger.info(f"  {folder_name}: {count} 个文件")
            
        return type_counters


    async def _process_multi_question_query_async(self, query: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
        """异步处理多问题查询"""
        start_time = time.time()
        
        questions = query['questions']
        original_images = query['images']
        force_type = query.get('force_type')
        metadata = query.get('metadata', {})
        
        logger.debug(f"处理多问题查询 {index}: {len(questions)} 个问题")
        
        try:
            # 等待图像预处理完成
            processed_images = []
            tif_results = []
            
            for img_path in original_images:
                processed_result = self.preprocessing_pipeline.get_processed_result(
                    img_path, 
                    timeout=600.0
                )
                
                if isinstance(processed_result, TifProcessResult):
                    processed_images.append(processed_result.resized_image_path)
                    tif_results.append(processed_result)
                    logger.debug(f"TIF图像处理完成: {img_path} -> {len(processed_result.tiles)} 个切片")
                else:
                    processed_images.append(str(processed_result))
                    tif_results.append(None)
            
            # 调用多问题处理方法
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.processor.process_multi_questions,
                questions,
                processed_images,
                force_type
            )
            
            # 为每个结果添加元数据和TIF信息
            for i, result in enumerate(results):
                extended_metadata = {**metadata}
                
                # 添加问题特定的元数据
                all_questions_data = metadata.get('all_questions_data', [])
                if i < len(all_questions_data):
                    extended_metadata.update({
                        'question_id': all_questions_data[i]['question_id'],
                        'original_content': all_questions_data[i]
                    })
                
                # 添加TIF处理信息
                tif_info = []
                for j, tif_result in enumerate(tif_results):
                    if tif_result:
                        tif_info.append({
                            'original_path': original_images[j],
                            'resized_path': tif_result.resized_image_path,
                            'tiles_count': len(tif_result.tiles),
                            'processing_time': tif_result.processing_time,
                            'json_path': tif_result.json_path
                        })
                
                result.update({
                    'metadata': extended_metadata,
                    'tif_processing_info': tif_info,
                    'original_images': original_images,
                    'processed_images': processed_images
                })
                
                # 记录对话
                if self.batch_config.save_conversations:
                    conversation = ConversationRecord(
                        timestamp=datetime.now().isoformat(),
                        question=result['question'],
                        question_type=result.get('question_type', ''),
                        images=original_images,
                        processed_images=processed_images,
                        prompt_used="",  # 多问题处理时的prompt
                        response=result.get('answer', ''),
                        success=result.get('success', False),
                        error=result.get('error'),
                        usage=result.get('usage'),
                        processing_time=result.get('processing_time', 0.0),
                        metadata=extended_metadata
                    )
                    self.conversations.append(conversation)
            
            logger.info(f"多问题查询 {index} 处理完成，共 {len(results)} 个问题")
            return results
            
        except Exception as e:
            logger.error(f"多问题查询 {index} 处理失败: {e}")
            processing_time = time.time() - start_time
            
            # 为所有问题创建错误结果
            error_results = []
            for i, question in enumerate(questions):
                extended_metadata = {**metadata}
                all_questions_data = metadata.get('all_questions_data', [])
                if i < len(all_questions_data):
                    extended_metadata.update({
                        'question_id': all_questions_data[i]['question_id'],
                        'original_content': all_questions_data[i]
                    })
                
                error_result = {
                    'success': False,
                    'question': question,
                    'error': str(e),
                    'question_index': i,
                    'processing_time': processing_time,
                    'metadata': extended_metadata,
                    'original_images': original_images
                }
                error_results.append(error_result)
            
            return error_results


def create_async_processor_from_config(config_path: str = "config/config.yaml") -> AsyncBatchProcessor:
    """从配置文件创建异步批量处理器"""
    return AsyncBatchProcessor(config_path)
