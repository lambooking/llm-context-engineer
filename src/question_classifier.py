"""
问题分类器模块
"""

import re
from typing import Optional
from .constants import QuestionType


class QuestionClassifier:
    """问题分类器，根据问题内容和图像数量判断问题类型"""
    
    def __init__(self):
        # 关键词模式匹配
        self.patterns = {
            QuestionType.IMAGE_DESC: [
                r'描述.*图',
                r'图.*描述',
                r'看到.*什么',
                r'图片.*内容',
                r'画面.*内容',
                r'这是.*图',
                r'图像.*显示'
            ],
            QuestionType.DUAL_IMAGE_COMPARE: [
                r'对比',
                r'比较',
                r'差异',
                r'不同',
                r'相同',
                r'区别',
                r'异同',
                r'两.*图',
                r'左右.*图'
            ],
            QuestionType.CHART_ANALYSIS: [
                r'图表',
                r'柱状图',
                r'折线图',
                r'饼图',
                r'散点图',
                r'计算',
                r'数据',
                r'统计',
                r'分析',
                r'趋势',
                r'百分比',
                r'数值',
                r'图形.*分析'
            ]
        }
    
    def classify(self, question: str, image_count: int = 1) -> QuestionType:
        """
        分类问题类型
        
        Args:
            question: 用户问题
            image_count: 图像数量
            
        Returns:
            QuestionType: 分类结果
        """
        question = question.lower().strip()
        
        # 如果有多张图片，优先考虑对比类问题
        if image_count > 1:
            if self._match_patterns(question, QuestionType.DUAL_IMAGE_COMPARE):
                return QuestionType.DUAL_IMAGE_COMPARE
        
        # 按优先级检查各类型
        for question_type in [QuestionType.CHART_ANALYSIS, 
                             QuestionType.IMAGE_DESC, 
                             QuestionType.DUAL_IMAGE_COMPARE]:
            if self._match_patterns(question, question_type):
                return question_type
        
        # 默认返回基础问答
        return QuestionType.BASIC_QA
    
    def _match_patterns(self, question: str, question_type: QuestionType) -> bool:
        """检查问题是否匹配指定类型的模式"""
        if question_type not in self.patterns:
            return False
            
        for pattern in self.patterns[question_type]:
            if re.search(pattern, question):
                return True
        return False
    
    def get_confidence(self, question: str, question_type: QuestionType) -> float:
        """
        获取分类置信度
        
        Args:
            question: 用户问题
            question_type: 问题类型
            
        Returns:
            float: 置信度 (0-1)
        """
        if question_type not in self.patterns:
            return 0.0
            
        matches = 0
        total_patterns = len(self.patterns[question_type])
        
        for pattern in self.patterns[question_type]:
            if re.search(pattern, question.lower()):
                matches += 1
        
        return matches / total_patterns if total_patterns > 0 else 0.0
