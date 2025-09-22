"""
Prompt管理模块 - 支持XML格式的Prompt加载和管理
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional
from pathlib import Path
from loguru import logger

from .constants import QuestionType, PROMPTS_DIR


class PromptManager:
    """XML格式的Prompt管理器"""
    
    def __init__(self, prompts_dir: str = PROMPTS_DIR):
        self.prompts_dir = Path(prompts_dir)
        self.prompts: Dict[QuestionType, str] = {}
        self.load_prompts()
    
    def load_prompts(self):
        """从XML文件加载所有Prompt"""
        try:
            # 确保prompts目录存在
            self.prompts_dir.mkdir(exist_ok=True)
            
            # 为每种问题类型加载prompt
            for question_type in QuestionType:
                self._load_prompt_for_type(question_type)
                
            logger.info(f"成功加载 {len(self.prompts)} 个Prompt模板")
            
        except Exception as e:
            logger.error(f"加载Prompt失败: {e}")
            self._create_default_prompts()
    
    def _load_prompt_for_type(self, question_type: QuestionType):
        """为特定类型加载Prompt"""
        filename = f"{question_type.name.lower()}.xml"
        filepath = self.prompts_dir / filename
        
        if filepath.exists():
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()
                
                # 查找prompt标签
                prompt_element = root.find('prompt')
                if prompt_element is not None:
                    # 使用itertext聚合文本，兼容嵌套标签与换行
                    prompt_text = ''.join(prompt_element.itertext()).strip() if prompt_element is not None else ''
                    if prompt_text:
                        self.prompts[question_type] = prompt_text
                        logger.info(f"从XML加载Prompt: {filepath}")
                    else:
                        logger.warning(f"XML文件 {filename} 的<prompt>内容为空")
                        self._create_default_prompt(question_type)
                else:
                    logger.warning(f"XML文件 {filename} 中未找到prompt标签")
                    self._create_default_prompt(question_type)
                    
            except ET.ParseError as e:
                logger.error(f"解析XML文件 {filename} 失败: {e}")
                self._create_default_prompt(question_type)
        else:
            logger.info(f"创建默认Prompt文件: {filename}")
            self._create_default_prompt(question_type)
    
    def _create_default_prompts(self):
        """创建所有默认Prompt"""
        for question_type in QuestionType:
            self._create_default_prompt(question_type)
    
    def _create_default_prompt(self, question_type: QuestionType):
        """创建并保存默认Prompt"""
        default_prompts = {
            QuestionType.BASIC_QA: """你是一个专业的AI助手，请根据提供的图像内容回答用户的问题。
请仔细观察图像中的细节，提供准确、详细的答案。

用户问题：{question}

请根据图像内容给出专业的回答。""",
            
            QuestionType.IMAGE_DESC: """请详细描述这张图像的内容。包括：
1. 主要物体和场景
2. 颜色、形状、位置等视觉特征
3. 任何可见的文字或标识
4. 整体的氛围和风格

请用清晰、有条理的语言进行描述。""",
            
            QuestionType.DUAL_IMAGE_COMPARE: """请仔细比较这些图像，并回答用户的问题。

比较要点：
1. 相同之处
2. 不同之处
3. 各自的特点
4. 关键差异分析

用户问题：{question}

请基于图像对比分析给出详细回答。""",
            
            QuestionType.CHART_ANALYSIS: """请分析这个图表并回答相关问题。

分析要点：
1. 图表类型和结构
2. 数据趋势和模式
3. 关键数值和计算
4. 结论和洞察

用户问题：{question}

请提供专业的数据分析和计算结果。"""
        }
        
        prompt_text = default_prompts.get(question_type, "请回答用户的问题。")
        self.prompts[question_type] = prompt_text
        
        # 保存到XML文件
        self._save_prompt_to_xml(question_type, prompt_text)
    
    def _save_prompt_to_xml(self, question_type: QuestionType, prompt_text: str):
        """将Prompt保存到XML文件"""
        try:
            # 创建XML结构
            root = ET.Element("prompt_template")
            root.set("type", question_type.value)
            root.set("name", question_type.name)
            
            # 添加描述
            desc = ET.SubElement(root, "description")
            desc.text = f"用于{question_type.value}的Prompt模板"
            
            # 添加prompt内容
            prompt_elem = ET.SubElement(root, "prompt")
            prompt_elem.text = prompt_text
            
            # 添加参数说明
            params = ET.SubElement(root, "parameters")
            if "{question}" in prompt_text:
                param = ET.SubElement(params, "parameter")
                param.set("name", "question")
                param.text = "用户问题"
            
            # 保存文件
            tree = ET.ElementTree(root)
            filename = f"{question_type.name.lower()}.xml"
            filepath = self.prompts_dir / filename
            
            tree.write(filepath, encoding='utf-8', xml_declaration=True)
            logger.debug(f"保存Prompt到文件: {filepath}")
            
        except Exception as e:
            logger.error(f"保存Prompt到XML失败: {e}")
    
    def get_prompt(self, question_type: QuestionType, **kwargs) -> str:
        """
        获取指定类型的Prompt
        
        Args:
            question_type: 问题类型
            **kwargs: 用于格式化Prompt的参数
            
        Returns:
            str: 格式化后的Prompt
        """
        prompt_template = self.prompts.get(question_type)
        if not prompt_template:
            logger.warning(f"未找到类型 {question_type.value} 的Prompt")
            # 基础回退：尽量包含传入的问题，避免信息缺失
            fallback = "请回答用户的问题。"
            question_text = kwargs.get("question")
            if question_text:
                return f"{fallback}\n\n{question_text}"
            return fallback
        
        formatted = prompt_template
        appended_question = False
        try:
            # 正常格式化（若模板包含 {question} 等占位）
            formatted = prompt_template.format(**kwargs)
        except KeyError as e:
            # 缺少参数时，降级为不格式化的模板
            logger.warning(f"Prompt格式化失败，缺少参数: {e}")
            formatted = prompt_template
        
        # 仅对基础QA类型进行“自动附加问题文本”，其它类型严格使用XML模板
        question_text = kwargs.get("question")
        try:
            from .constants import QuestionType as _QT
            is_basic_qa = question_type == _QT.BASIC_QA
        except Exception:
            is_basic_qa = False
        if is_basic_qa and question_text and ("{question}" not in prompt_template) and (question_text not in formatted):
            formatted = f"{formatted}\n\n{question_text}"
            appended_question = True
        
        if appended_question:
            logger.debug("Prompt模板未包含{question}，已附加传入问题到Prompt末尾")
        
        return formatted
    
    def update_prompt(self, question_type: QuestionType, new_prompt: str):
        """
        更新指定类型的Prompt
        
        Args:
            question_type: 问题类型
            new_prompt: 新的Prompt内容
        """
        self.prompts[question_type] = new_prompt
        self._save_prompt_to_xml(question_type, new_prompt)
        logger.info(f"更新Prompt: {question_type.value}")
    
    def reload_prompts(self):
        """重新加载所有Prompt"""
        self.prompts.clear()
        self.load_prompts()
        logger.info("重新加载所有Prompt完成")
