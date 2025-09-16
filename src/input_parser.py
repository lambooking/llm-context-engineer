"""
结构化输入格式解析器
支持处理按照特定目录结构组织的输入数据
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .constants import QuestionType


@dataclass
class ParsedInput:
    """解析后的输入数据结构"""
    question_type: QuestionType
    questions: List[str]
    images: List[List[str]]  # 每个问题对应的图像列表
    metadata: Dict[str, Any]


class InputParser:
    """结构化输入格式解析器"""
    
    def __init__(self):
        # 支持的图像格式
        self.image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
        # 支持的文本格式
        self.text_extensions = {'.txt', '.md'}
        
        # 目录类型映射
        self.directory_type_mapping = {
            'qa': QuestionType.BASIC_QA,
            'image_caption': QuestionType.IMAGE_DESC,
            'change_caption': QuestionType.DUAL_IMAGE_COMPARE,
            'chart_analysis': QuestionType.CHART_ANALYSIS
        }
    
    def parse_input_directory(self, input_path: str) -> List[ParsedInput]:
        """
        解析输入目录结构
        
        Args:
            input_path: 输入路径根目录
            
        Returns:
            List[ParsedInput]: 解析结果列表
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入路径不存在: {input_path}")
        
        results = []
        
        # 遍历顶层目录
        for type_dir in input_path.iterdir():
            if type_dir.is_dir():
                logger.info(f"处理类型目录: {type_dir.name}")
                
                # 确定问题类型
                question_type = self._determine_question_type(type_dir.name.lower())
                if question_type is None:
                    logger.warning(f"未识别的目录类型: {type_dir.name}")
                    continue
                
                # 根据类型解析目录
                parsed_data = self._parse_type_directory(type_dir, question_type)
                if parsed_data:
                    results.extend(parsed_data)
        
        logger.info(f"总共解析出 {len(results)} 个输入项")
        return results
    
    def _determine_question_type(self, dir_name: str) -> Optional[QuestionType]:
        """根据目录名确定问题类型"""
        return self.directory_type_mapping.get(dir_name)
    
    def _parse_type_directory(self, type_dir: Path, question_type: QuestionType) -> List[ParsedInput]:
        """解析特定类型的目录"""
        if question_type == QuestionType.DUAL_IMAGE_COMPARE:
            return self._parse_change_caption_directory(type_dir, question_type)
        else:
            return self._parse_standard_directory(type_dir, question_type)
    
    def _parse_standard_directory(self, type_dir: Path, question_type: QuestionType) -> List[ParsedInput]:
        """
        解析标准目录结构 (QA/, Image_caption/)
        结构: type_dir/image/, type_dir/question/
        """
        results = []
        
        image_dir = type_dir / "image"
        question_dir = type_dir / "question"
        
        if not image_dir.exists() or not question_dir.exists():
            logger.warning(f"目录结构不完整: {type_dir}")
            return results
        
        # 获取问题文件和图像文件的映射
        question_files = self._get_numbered_files(question_dir, self.text_extensions)
        image_files = self._get_numbered_files(image_dir, self.image_extensions)
        
        # 按编号配对
        for number in sorted(set(question_files.keys()) & set(image_files.keys())):
            try:
                # 解析多问题文件
                questions_data = self._parse_multi_question_file(question_files[number])
                if not questions_data:
                    logger.warning(f"问题文件解析失败或为空: {question_files[number]}")
                    continue
                
                # 获取对应的图像
                images = [str(image_files[number])]
                
                # 将同一文件的所有问题组合成一个ParsedInput（优化：一次处理多个问题）
                all_questions = [q_data['text_input'] for q_data in questions_data]
                
                parsed_input = ParsedInput(
                    question_type=question_type,
                    questions=all_questions,  # 包含所有问题
                    images=[images],
                    metadata={
                        'source_dir': str(type_dir),
                        'file_number': number,
                        'original_file': str(question_files[number]),
                        'image_files': images,
                        'all_questions_data': questions_data,  # 保存所有问题数据
                        'is_multi_question': True  # 标记为多问题
                    }
                )
                
                results.append(parsed_input)
                logger.debug(f"解析多问题文件: {type_dir.name}/{number} ({len(questions_data)} 个问题)")
                
            except Exception as e:
                logger.error(f"解析失败 {type_dir.name}/{number}: {e}")
                continue
        
        return results
    
    def _parse_change_caption_directory(self, type_dir: Path, question_type: QuestionType) -> List[ParsedInput]:
        """
        解析变化描述目录结构 (Change_caption/)
        结构: type_dir/image1/, type_dir/image2/, type_dir/question/
        """
        results = []
        
        image1_dir = type_dir / "image1"
        image2_dir = type_dir / "image2"
        question_dir = type_dir / "question"
        
        if not all([image1_dir.exists(), image2_dir.exists(), question_dir.exists()]):
            logger.warning(f"Change_caption目录结构不完整: {type_dir}")
            return results
        
        # 获取文件映射
        question_files = self._get_numbered_files(question_dir, self.text_extensions)
        image1_files = self._get_numbered_files(image1_dir, self.image_extensions)
        image2_files = self._get_numbered_files(image2_dir, self.image_extensions)
        
        # 按编号配对
        common_numbers = set(question_files.keys()) & set(image1_files.keys()) & set(image2_files.keys())
        
        for number in sorted(common_numbers):
            try:
                # 解析多问题文件
                questions_data = self._parse_multi_question_file(question_files[number])
                if not questions_data:
                    logger.warning(f"问题文件解析失败或为空: {question_files[number]}")
                    continue
                
                # 获取对应的图像对
                images = [str(image1_files[number]), str(image2_files[number])]
                
                # 将同一文件的所有问题组合成一个ParsedInput（优化：一次处理多个问题）
                all_questions = [q_data['text_input'] for q_data in questions_data]
                
                parsed_input = ParsedInput(
                    question_type=question_type,
                    questions=all_questions,  # 包含所有问题
                    images=[images],
                    metadata={
                        'source_dir': str(type_dir),
                        'file_number': number,
                        'original_file': str(question_files[number]),
                        'image_files': images,
                        'all_questions_data': questions_data,  # 保存所有问题数据
                        'is_multi_question': True  # 标记为多问题
                    }
                )
                
                results.append(parsed_input)
                logger.debug(f"解析多问题文件: {type_dir.name}/{number} ({len(questions_data)} 个问题)")
                
            except Exception as e:
                logger.error(f"解析失败 {type_dir.name}/{number}: {e}")
                continue
        
        return results
    
    def _get_numbered_files(self, directory: Path, extensions: set) -> Dict[int, Path]:
        """
        获取目录中按编号排序的文件
        
        Args:
            directory: 目录路径
            extensions: 允许的文件扩展名
            
        Returns:
            Dict[int, Path]: 编号到文件路径的映射
        """
        files = {}
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    # 从文件名提取编号
                    number = int(file_path.stem)
                    files[number] = file_path
                except ValueError:
                    logger.warning(f"无法从文件名提取编号: {file_path}")
                    continue
        
        return files
    
    def _read_text_file(self, file_path: Path) -> str:
        """读取文本文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read().strip()
    
    def _parse_multi_question_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        解析多问题格式的文本文件
        格式：
        image_path: ......./0.tif
        question_id: 0
        text_input: 问题内容
        text_truth:
        question_id: 1
        text_input: 另一个问题
        text_truth:
        ...
        """
        content = self._read_text_file(file_path)
        questions = []
        
        lines = content.split('\n')
        current_question = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('image_path:'):
                current_question['image_path'] = line.split(':', 1)[1].strip()
            elif line.startswith('question_id:'):
                # 如果已经有问题在处理，保存它
                if 'question_id' in current_question and 'text_input' in current_question:
                    questions.append(current_question.copy())
                
                current_question['question_id'] = line.split(':', 1)[1].strip()
            elif line.startswith('text_input:'):
                current_question['text_input'] = line.split(':', 1)[1].strip()
            elif line.startswith('text_truth:'):
                current_question['text_truth'] = line.split(':', 1)[1].strip() if ':' in line and len(line.split(':', 1)) > 1 else ''
        
        # 添加最后一个问题
        if 'question_id' in current_question and 'text_input' in current_question:
            questions.append(current_question)
        
        return questions
    
    def convert_to_queries(self, parsed_inputs: List[ParsedInput]) -> List[Dict[str, Any]]:
        """
        将解析结果转换为处理器可用的查询格式
        
        Args:
            parsed_inputs: 解析结果列表
            
        Returns:
            List[Dict]: 查询格式列表
        """
        queries = []
        
        for parsed_input in parsed_inputs:
            if parsed_input.metadata.get('is_multi_question', False):
                # 多问题情况：创建一个包含所有问题的查询
                query = {
                    'questions': parsed_input.questions,  # 多个问题
                    'images': parsed_input.images[0],  # 图像列表
                    'force_type': parsed_input.question_type,
                    'metadata': parsed_input.metadata,
                    'is_multi_question': True
                }
                queries.append(query)
            else:
                # 单问题情况：保持原有逻辑
                for i, (question, images) in enumerate(zip(parsed_input.questions, parsed_input.images)):
                    query = {
                        'question': question,
                        'images': images,
                        'force_type': parsed_input.question_type,
                        'metadata': {
                            **parsed_input.metadata,
                            'sub_index': i
                        }
                    }
                    queries.append(query)
        
        return queries
    
    def validate_input_structure(self, input_path: str) -> Dict[str, Any]:
        """
        验证输入目录结构
        
        Args:
            input_path: 输入路径
            
        Returns:
            Dict: 验证结果
        """
        input_path = Path(input_path)
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        if not input_path.exists():
            result['valid'] = False
            result['errors'].append(f"输入路径不存在: {input_path}")
            return result
        
        # 检查每个类型目录
        for type_dir in input_path.iterdir():
            if type_dir.is_dir():
                type_name = type_dir.name.lower()
                
                if type_name not in self.directory_type_mapping:
                    result['warnings'].append(f"未识别的目录类型: {type_dir.name}")
                    continue
                
                # 验证目录结构
                if type_name == 'change_caption':
                    required_subdirs = ['image1', 'image2', 'question']
                else:
                    required_subdirs = ['image', 'question']
                
                missing_dirs = []
                for subdir in required_subdirs:
                    if not (type_dir / subdir).exists():
                        missing_dirs.append(subdir)
                
                if missing_dirs:
                    result['errors'].append(f"{type_dir.name} 缺少目录: {missing_dirs}")
                    result['valid'] = False
                
                # 统计文件数量
                result['summary'][type_dir.name] = self._count_files_in_type_dir(type_dir)
        
        return result
    
    def _count_files_in_type_dir(self, type_dir: Path) -> Dict[str, int]:
        """统计类型目录中的文件数量"""
        counts = {}
        
        for subdir in type_dir.iterdir():
            if subdir.is_dir():
                file_count = len([f for f in subdir.iterdir() if f.is_file()])
                counts[subdir.name] = file_count
        
        return counts
