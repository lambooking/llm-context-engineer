#!/usr/bin/env python3
"""
问题分类器测试
"""

import unittest
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.question_classifier import QuestionClassifier
from src.constants import QuestionType


class TestQuestionClassifier(unittest.TestCase):
    """问题分类器测试类"""
    
    def setUp(self):
        """测试准备"""
        self.classifier = QuestionClassifier()
    
    def test_image_description_classification(self):
        """测试图像描述分类"""
        questions = [
            "描述这张图片",
            "这张图像显示了什么",
            "图片内容是什么",
            "看到了什么画面"
        ]
        
        for question in questions:
            result = self.classifier.classify(question)
            self.assertEqual(result, QuestionType.IMAGE_DESC, 
                           f"问题 '{question}' 应该被分类为图像描述")
    
    def test_dual_image_compare_classification(self):
        """测试对比分析分类"""
        questions = [
            "比较这两张图片的差异",
            "对比左右两张图",
            "这两个图像有什么不同",
            "分析图片的异同点"
        ]
        
        for question in questions:
            # 单图情况
            result = self.classifier.classify(question, 1)
            self.assertEqual(result, QuestionType.DUAL_IMAGE_COMPARE)
            
            # 多图情况
            result = self.classifier.classify(question, 2)
            self.assertEqual(result, QuestionType.DUAL_IMAGE_COMPARE)
    
    def test_chart_analysis_classification(self):
        """测试图表分析分类"""
        questions = [
            "分析这个图表的数据",
            "计算柱状图中的数值",
            "这个饼图显示什么趋势",
            "统计图形中的百分比"
        ]
        
        for question in questions:
            result = self.classifier.classify(question)
            self.assertEqual(result, QuestionType.CHART_ANALYSIS,
                           f"问题 '{question}' 应该被分类为图表分析")
    
    def test_basic_qa_classification(self):
        """测试基础问答分类"""
        questions = [
            "这是什么东西",
            "帮我解答这个问题",
            "请告诉我答案",
            "这个怎么理解"
        ]
        
        for question in questions:
            result = self.classifier.classify(question)
            self.assertEqual(result, QuestionType.BASIC_QA,
                           f"问题 '{question}' 应该被分类为基础问答")
    
    def test_multi_image_priority(self):
        """测试多图像优先级"""
        question = "比较这些图片"
        
        # 多图像时应该优先考虑对比类型
        result = self.classifier.classify(question, 3)
        self.assertEqual(result, QuestionType.DUAL_IMAGE_COMPARE)
    
    def test_confidence_calculation(self):
        """测试置信度计算"""
        question = "描述图像内容并分析数据趋势"
        
        # 获取各类型的置信度
        desc_confidence = self.classifier.get_confidence(question, QuestionType.IMAGE_DESC)
        chart_confidence = self.classifier.get_confidence(question, QuestionType.CHART_ANALYSIS)
        
        # 应该都有一定的置信度
        self.assertGreater(desc_confidence, 0)
        self.assertGreater(chart_confidence, 0)
        
        # 置信度应该在0-1之间
        self.assertLessEqual(desc_confidence, 1.0)
        self.assertLessEqual(chart_confidence, 1.0)
    
    def test_pattern_matching(self):
        """测试模式匹配"""
        # 测试私有方法
        question = "图表分析"
        
        # 应该匹配图表分析模式
        result = self.classifier._match_patterns(question, QuestionType.CHART_ANALYSIS)
        self.assertTrue(result)
        
        # 不应该匹配图像描述模式
        result = self.classifier._match_patterns(question, QuestionType.IMAGE_DESC)
        self.assertFalse(result)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空字符串
        result = self.classifier.classify("")
        self.assertEqual(result, QuestionType.BASIC_QA)
        
        # 只有空格
        result = self.classifier.classify("   ")
        self.assertEqual(result, QuestionType.BASIC_QA)
        
        # 混合大小写
        result = self.classifier.classify("描述这张图片")
        self.assertEqual(result, QuestionType.IMAGE_DESC)
        
        # 零图像数量
        result = self.classifier.classify("比较图片", 0)
        self.assertEqual(result, QuestionType.DUAL_IMAGE_COMPARE)


if __name__ == '__main__':
    unittest.main()
