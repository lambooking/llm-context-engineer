"""
常量定义
"""

from enum import Enum

class QuestionType(Enum):
    """问题类型枚举"""
    BASIC_QA = "基础问答"
    IMAGE_DESC = "图像描述"
    DUAL_IMAGE_COMPARE = "对照对比问答类/差异描述"
    CHART_ANALYSIS = "图表分析/计算"

# API配置
DEFAULT_MODEL = "gpt-4-vision-preview"
MAX_TOKENS = 4000
TEMPERATURE = 0.7

# 文件路径
PROMPTS_DIR = "prompts"
CONFIG_DIR = "config"
