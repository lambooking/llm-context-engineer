"""
自定义VLM客户端 - 支持/generate端点格式
"""

import base64
import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from loguru import logger

from .constants import DEFAULT_MODEL, MAX_TOKENS, TEMPERATURE


class CustomVLMClient:
    """自定义VLM API客户端，支持/generate端点"""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:1237", model: str = DEFAULT_MODEL, timeout: int = 60):
        """
        初始化自定义VLM客户端
        
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
            'Content-Type': 'application/json'
        })
        
        # 如果有API密钥，添加授权头
        if api_key and api_key != "sk-12":  # 跳过占位符密钥
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
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
    
    def chat_completion(
        self, 
        prompt: str, 
        images: List[Union[str, Path]] = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        **kwargs
    ) -> Dict[str, Any]:
        """
        发送聊天完成请求到/generate端点
        
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
            # 构建请求体 - 适配/generate端点格式
            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
            }
            
            # 添加图像（如果有）
            if images:
                image_data_list = []
                for image_path in images:
                    try:
                        image_data = self.encode_image(image_path)
                        image_data_list.append(image_data)
                    except Exception as e:
                        logger.warning(f"跳过无法编码的图像 {image_path}: {e}")
                        continue
                
                if image_data_list:
                    payload["image"] = image_data_list[0] if len(image_data_list) == 1 else image_data_list
            
            logger.debug(f"发送请求到 {self.base_url}/generate")
            
            # 发送请求
            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout  # 使用配置的超时时间
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 转换响应格式以兼容原有代码
            if "text" in result:
                # 转换为OpenAI格式的响应
                converted_result = {
                    "choices": [
                        {
                            "message": {
                                "content": result["text"]
                            }
                        }
                    ],
                    "usage": result.get("meta_info", {}).get("usage", {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    })
                }
                
                logger.info(f"自定义VLM API调用成功，模型: {self.model}")
                return converted_result
            else:
                logger.error(f"意外的响应格式: {result}")
                raise ValueError(f"意外的响应格式: {result}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"自定义VLM API请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"自定义VLM API调用异常: {e}")
            raise
    
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


class EnhancedVLMClientFactory:
    """增强的VLM客户端工厂类"""
    
    @staticmethod
    def create_client(provider: str, **config):
        """
        创建VLM客户端实例
        
        Args:
            provider: 服务提供商 (openai, azure, custom)
            **config: 配置参数
            
        Returns:
            VLMClient: 客户端实例
        """
        base_url = config.get('base_url', 'https://api.openai.com/v1')
        
        # 对于localhost的OpenAI兼容API，使用标准客户端
        from .vlm_client import VLMClient
        logger.info(f"使用标准VLM客户端连接到: {base_url}")
        return VLMClient(
            api_key=config.get('api_key', 'sk-12'),  # 本地API可能不需要真实密钥
            base_url=base_url,
            model=config.get('model', DEFAULT_MODEL),
            timeout=config.get('timeout', 60)  # 传递超时配置
        )
