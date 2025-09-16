"""
VLM API客户端模块 - 支持GPT-4V格式的API调用
"""

import base64
import requests
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from loguru import logger

from .constants import DEFAULT_MODEL, MAX_TOKENS, TEMPERATURE


class VLMClient:
    """VLM API客户端，支持GPT-4V格式的请求"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = DEFAULT_MODEL, timeout: int = 60):
        """
        初始化VLM客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 使用的模型名称
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def encode_image(self, image_path: Union[str, Path]) -> str:
        """
        将图像编码为base64格式
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: base64编码的图像数据
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"图像编码失败 {image_path}: {e}")
            raise
    
    def create_image_message(self, image_data: str, image_type: str = "image/jpeg") -> Dict[str, Any]:
        """
        创建图像消息格式
        
        Args:
            image_data: base64编码的图像数据
            image_type: 图像MIME类型
            
        Returns:
            Dict: 图像消息格式
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_type};base64,{image_data}"
            }
        }
    
    def create_text_message(self, text: str) -> Dict[str, Any]:
        """
        创建文本消息格式
        
        Args:
            text: 文本内容
            
        Returns:
            Dict: 文本消息格式
        """
        return {
            "type": "text",
            "text": text
        }
    
    def chat_completion(
        self, 
        prompt: str, 
        images: List[Union[str, Path]] = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        **kwargs
    ) -> Dict[str, Any]:
        """
        发送聊天完成请求
        
        Args:
            prompt: 文本提示
            images: 图像文件路径列表
            max_tokens: 最大token数
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            Dict: API响应
        """
        try:
            # 构建消息内容
            content = [self.create_text_message(prompt)]
            
            # 添加图像
            if images:
                for image_path in images:
                    # 检测图像类型
                    image_type = self._detect_image_type(image_path)
                    # 编码图像
                    image_data = self.encode_image(image_path)
                    # 添加到消息内容
                    content.append(self.create_image_message(image_data, image_type))
            
            # 构建请求体
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            # 发送请求
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"VLM API调用成功，模型: {self.model}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"VLM API请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"VLM API调用异常: {e}")
            raise
    
    def _detect_image_type(self, image_path: Union[str, Path]) -> str:
        """
        检测图像文件类型
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: MIME类型
        """
        path = Path(image_path)
        suffix = path.suffix.lower()
        
        type_mapping = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        
        return type_mapping.get(suffix, 'image/jpeg')
    
    def extract_response_text(self, response: Dict[str, Any]) -> str:
        """
        从API响应中提取文本内容
        
        Args:
            response: API响应
            
        Returns:
            str: 提取的文本内容
        """
        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            logger.error(f"解析API响应失败: {e}")
            return "抱歉，无法解析API响应。"
    
    def get_usage_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        """
        获取API使用情况信息
        
        Args:
            response: API响应
            
        Returns:
            Dict: 使用情况信息
        """
        try:
            usage = response.get('usage', {})
            return {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        except Exception as e:
            logger.error(f"获取使用情况信息失败: {e}")
            return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


class VLMClientFactory:
    """VLM客户端工厂类"""
    
    @staticmethod
    def create_client(provider: str, **config) -> VLMClient:
        """
        创建VLM客户端实例
        
        Args:
            provider: 服务提供商 (openai, azure, etc.)
            **config: 配置参数
            
        Returns:
            VLMClient: 客户端实例
        """
        if provider.lower() == 'openai':
            return VLMClient(
                api_key=config['api_key'],
                base_url=config.get('base_url', 'https://api.openai.com/v1'),
                model=config.get('model', DEFAULT_MODEL)
            )
        elif provider.lower() == 'azure':
            # Azure OpenAI配置
            return VLMClient(
                api_key=config['api_key'],
                base_url=config['base_url'],
                model=config.get('model', DEFAULT_MODEL)
            )
        else:
            # 通用配置
            return VLMClient(
                api_key=config['api_key'],
                base_url=config.get('base_url', 'https://api.openai.com/v1'),
                model=config.get('model', DEFAULT_MODEL)
            )
