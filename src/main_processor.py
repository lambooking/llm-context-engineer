"""
主处理器模块 - 整合所有组件的核心处理流程
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from loguru import logger

from .constants import QuestionType
from .question_classifier import QuestionClassifier
from .prompt_manager import PromptManager
from .vlm_client import VLMClient, VLMClientFactory


class LLMContextProcessor:
    """LLM上下文工程处理器 - 核心处理类"""
    
    def __init__(self, vlm_client: VLMClient, prompts_dir: str = "prompts"):
        """
        初始化处理器
        
        Args:
            vlm_client: VLM客户端实例
            prompts_dir: Prompt文件目录
        """
        self.vlm_client = vlm_client
        self.classifier = QuestionClassifier()
        self.prompt_manager = PromptManager(prompts_dir)
        
        logger.info("LLM上下文处理器初始化完成")
    
    def process_query(
        self, 
        question: str, 
        images: List[Union[str, Path]] = None,
        force_type: Optional[QuestionType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理用户查询的主要方法
        
        Args:
            question: 用户问题
            images: 图像文件路径列表
            force_type: 强制指定问题类型（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict: 处理结果
        """
        try:
            # 1. 问题分类
            if force_type:
                question_type = force_type
                logger.info(f"使用强制指定的问题类型: {question_type.value}")
            else:
                image_count = len(images) if images else 0
                question_type = self.classifier.classify(question, image_count)
                confidence = self.classifier.get_confidence(question, question_type)
                logger.info(f"问题分类结果: {question_type.value} (置信度: {confidence:.2f})")
            
            # 2. 获取对应的Prompt
            prompt = self.prompt_manager.get_prompt(question_type, question=question)
            logger.debug(f"使用Prompt模板: {question_type.value}")
            
            # 3. 调用VLM API
            response = self.vlm_client.chat_completion(
                prompt=prompt,
                images=images,
                **kwargs
            )
            
            # 4. 提取响应内容
            answer = self.vlm_client.extract_response_text(response)
            usage_info = self.vlm_client.get_usage_info(response)
            
            # 5. 构建结果
            result = {
                'success': True,
                'question': question,
                'question_type': question_type.value,
                'answer': answer,
                'images_count': len(images) if images else 0,
                'usage': usage_info,
                'raw_response': response
            }
            
            if not force_type:
                result['classification_confidence'] = confidence
            
            logger.info("查询处理成功完成")
            return result
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            return {
                'success': False,
                'question': question,
                'error': str(e),
                'images_count': len(images) if images else 0
            }
    
    def batch_process(
        self, 
        queries: List[Dict[str, Any]], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量处理查询
        
        Args:
            queries: 查询列表，每个查询包含question和images字段
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: 处理结果列表
        """
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"处理批量查询 {i+1}/{len(queries)}")
            
            question = query.get('question', '')
            images = query.get('images', [])
            force_type = query.get('force_type')
            
            result = self.process_query(
                question=question,
                images=images,
                force_type=force_type,
                **kwargs
            )
            
            result['batch_index'] = i
            results.append(result)
        
        logger.info(f"批量处理完成，共处理 {len(queries)} 个查询")
        return results
    
    def get_classification_preview(self, question: str, image_count: int = 1) -> Dict[str, Any]:
        """
        获取问题分类预览（不调用API）
        
        Args:
            question: 用户问题
            image_count: 图像数量
            
        Returns:
            Dict: 分类预览结果
        """
        question_type = self.classifier.classify(question, image_count)
        confidence = self.classifier.get_confidence(question, question_type)
        
        # 获取所有类型的置信度
        all_confidences = {}
        for qtype in QuestionType:
            all_confidences[qtype.value] = self.classifier.get_confidence(question, qtype)
        
        return {
            'question': question,
            'predicted_type': question_type.value,
            'confidence': confidence,
            'all_confidences': all_confidences,
            'image_count': image_count
        }
    
    def update_prompt(self, question_type: QuestionType, new_prompt: str):
        """
        更新指定类型的Prompt
        
        Args:
            question_type: 问题类型
            new_prompt: 新的Prompt内容
        """
        self.prompt_manager.update_prompt(question_type, new_prompt)
        logger.info(f"已更新 {question_type.value} 的Prompt模板")
    
    def reload_prompts(self):
        """重新加载所有Prompt模板"""
        self.prompt_manager.reload_prompts()
        logger.info("已重新加载所有Prompt模板")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取处理器统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'available_question_types': [qtype.value for qtype in QuestionType],
            'loaded_prompts': len(self.prompt_manager.prompts),
            'vlm_model': self.vlm_client.model,
            'vlm_base_url': self.vlm_client.base_url
        }


class ProcessorFactory:
    """处理器工厂类"""
    
    @staticmethod
    def create_processor(config: Dict[str, Any]) -> LLMContextProcessor:
        """
        根据配置创建处理器实例
        
        Args:
            config: 配置字典
            
        Returns:
            LLMContextProcessor: 处理器实例
        """
        # 创建VLM客户端
        vlm_config = config.get('vlm', {})
        provider = vlm_config.get('provider', 'openai')
        vlm_client = VLMClientFactory.create_client(provider, **vlm_config)
        
        # 创建处理器
        prompts_dir = config.get('prompts_dir', 'prompts')
        processor = LLMContextProcessor(vlm_client, prompts_dir)
        
        return processor
    
    @staticmethod
    def create_from_env() -> LLMContextProcessor:
        """
        从环境变量创建处理器实例
        
        Returns:
            LLMContextProcessor: 处理器实例
        """
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        config = {
            'vlm': {
                'provider': os.getenv('VLM_PROVIDER', 'openai'),
                'api_key': os.getenv('VLM_API_KEY'),
                'base_url': os.getenv('VLM_BASE_URL', 'https://api.openai.com/v1'),
                'model': os.getenv('VLM_MODEL', 'gpt-4-vision-preview')
            },
            'prompts_dir': os.getenv('PROMPTS_DIR', 'prompts')
        }
        
        return ProcessorFactory.create_processor(config)
