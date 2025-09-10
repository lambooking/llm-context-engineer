"""
图像预处理模块
专门处理遥感图像等大分辨率图像的预处理
"""

import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

from PIL import Image, ImageOps
from loguru import logger


@dataclass
class PreprocessConfig:
    """预处理配置"""
    max_resolution: Tuple[int, int] = (2048, 2048)
    quality: int = 85
    format: str = "JPEG"
    cache_dir: str = "cache/processed_images"
    parallel_workers: int = 4
    enabled: bool = True


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程锁，用于缓存操作
        self._cache_lock = threading.Lock()
        
        logger.info(f"图像预处理器初始化完成，缓存目录: {self.cache_dir}")
    
    def preprocess_image(self, image_path: str) -> str:
        """
        预处理单个图像
        
        Args:
            image_path: 原始图像路径
            
        Returns:
            str: 预处理后的图像路径
        """
        if not self.config.enabled:
            return image_path
        
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"图像文件不存在: {image_path}")
            return str(image_path)
        
        # 生成缓存文件名
        cache_filename = self._generate_cache_filename(image_path)
        cache_path = self.cache_dir / cache_filename
        
        # 检查缓存
        if cache_path.exists():
            logger.debug(f"使用缓存图像: {cache_path}")
            return str(cache_path)
        
        try:
            # 执行预处理
            processed_path = self._process_single_image(image_path, cache_path)
            logger.debug(f"图像预处理完成: {image_path} -> {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"图像预处理失败 {image_path}: {e}")
            return str(image_path)  # 返回原始路径作为fallback
    
    def preprocess_images_batch(self, image_paths: List[str]) -> List[str]:
        """
        批量预处理图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            List[str]: 预处理后的图像路径列表
        """
        if not self.config.enabled:
            return image_paths
        
        if not image_paths:
            return []
        
        logger.info(f"开始批量预处理 {len(image_paths)} 个图像")
        
        processed_paths = [None] * len(image_paths)
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # 提交任务
            future_to_index = {
                executor.submit(self.preprocess_image, path): i 
                for i, path in enumerate(image_paths)
            }
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    processed_path = future.result()
                    processed_paths[index] = processed_path
                except Exception as e:
                    logger.error(f"批量预处理失败 {image_paths[index]}: {e}")
                    processed_paths[index] = image_paths[index]
        
        logger.info(f"批量预处理完成")
        return processed_paths
    
    def _process_single_image(self, input_path: Path, output_path: Path) -> str:
        """处理单个图像文件"""
        with Image.open(input_path) as img:
            # 获取原始尺寸
            original_size = img.size
            logger.debug(f"原始图像尺寸: {original_size}")
            
            # 检查是否需要调整尺寸
            if self._needs_resize(original_size):
                img = self._resize_image(img)
                logger.debug(f"调整后尺寸: {img.size}")
            
            # 转换格式（如果需要）
            if img.mode not in ('RGB', 'L'):
                if img.mode == 'RGBA':
                    # 处理透明背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存处理后的图像
            save_kwargs = {
                'format': self.config.format,
                'optimize': True
            }
            
            if self.config.format.upper() == 'JPEG':
                save_kwargs['quality'] = self.config.quality
            
            with self._cache_lock:
                img.save(output_path, **save_kwargs)
            
            return str(output_path)
    
    def _needs_resize(self, size: Tuple[int, int]) -> bool:
        """检查是否需要调整尺寸"""
        width, height = size
        max_width, max_height = self.config.max_resolution
        return width > max_width or height > max_height
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """调整图像尺寸，保持宽高比"""
        max_width, max_height = self.config.max_resolution
        
        # 计算缩放比例
        width_ratio = max_width / img.width
        height_ratio = max_height / img.height
        ratio = min(width_ratio, height_ratio)
        
        # 计算新尺寸
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        
        # 使用高质量重采样
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _generate_cache_filename(self, image_path: Path) -> str:
        """生成缓存文件名"""
        # 使用文件路径和修改时间生成哈希
        stat = image_path.stat()
        content = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
        
        # 添加预处理参数到哈希中
        config_str = f"{self.config.max_resolution}_{self.config.quality}_{self.config.format}"
        content += f"_{config_str}"
        
        hash_obj = hashlib.md5(content.encode())
        hash_str = hash_obj.hexdigest()
        
        # 使用原始文件的扩展名（如果适用）
        if self.config.format.upper() == 'JPEG':
            ext = '.jpg'
        else:
            ext = f'.{self.config.format.lower()}'
        
        return f"{hash_str}{ext}"
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        if not self.cache_dir.exists():
            return {
                'cache_dir': str(self.cache_dir),
                'exists': False,
                'file_count': 0,
                'total_size': 0
            }
        
        cache_files = list(self.cache_dir.glob('*'))
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
        
        return {
            'cache_dir': str(self.cache_dir),
            'exists': True,
            'file_count': len(cache_files),
            'total_size': total_size,
            'total_size_mb': round(total_size / 1024 / 1024, 2)
        }
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        清理缓存
        
        Args:
            older_than_days: 清理多少天前的缓存，None表示清理所有
        """
        if not self.cache_dir.exists():
            return
        
        import time
        current_time = time.time()
        cutoff_time = current_time - (older_than_days * 24 * 3600) if older_than_days else 0
        
        removed_count = 0
        removed_size = 0
        
        for cache_file in self.cache_dir.glob('*'):
            if cache_file.is_file():
                if cache_file.stat().st_mtime < cutoff_time:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    removed_count += 1
                    removed_size += file_size
        
        logger.info(f"清理缓存完成: 删除 {removed_count} 个文件，释放 {removed_size/1024/1024:.2f} MB")


class PreprocessingPipeline:
    """预处理流水线"""
    
    def __init__(self, preprocessor: ImagePreprocessor):
        self.preprocessor = preprocessor
        self._processing_queue = []
        self._processed_cache = {}
    
    def add_to_pipeline(self, image_paths: List[str]) -> List[str]:
        """
        添加图像到预处理流水线
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            List[str]: 预处理后的路径列表（可能包含占位符）
        """
        # 检查缓存
        processed_paths = []
        to_process = []
        
        for path in image_paths:
            if path in self._processed_cache:
                processed_paths.append(self._processed_cache[path])
            else:
                # 添加到处理队列
                to_process.append(path)
                processed_paths.append(None)  # 占位符
        
        if to_process:
            # 启动异步预处理
            self._start_async_processing(to_process)
        
        return processed_paths
    
    def _start_async_processing(self, image_paths: List[str]):
        """启动异步预处理"""
        import threading
        
        def process_async():
            processed = self.preprocessor.preprocess_images_batch(image_paths)
            for original, processed_path in zip(image_paths, processed):
                self._processed_cache[original] = processed_path
        
        thread = threading.Thread(target=process_async)
        thread.daemon = True
        thread.start()
    
    def get_processed_path(self, original_path: str, timeout: float = 30.0) -> str:
        """
        获取预处理后的路径，等待处理完成
        
        Args:
            original_path: 原始路径
            timeout: 超时时间
            
        Returns:
            str: 预处理后的路径
        """
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if original_path in self._processed_cache:
                return self._processed_cache[original_path]
            time.sleep(0.1)
        
        # 超时，返回原始路径
        logger.warning(f"预处理超时，使用原始路径: {original_path}")
        return original_path


def create_preprocessor_from_config(config_dict: Dict[str, Any]) -> ImagePreprocessor:
    """从配置字典创建预处理器"""
    preprocess_config = config_dict.get('image_preprocessing', {})
    
    config = PreprocessConfig(
        max_resolution=tuple(preprocess_config.get('max_resolution', [2048, 2048])),
        quality=preprocess_config.get('quality', 85),
        format=preprocess_config.get('format', 'JPEG'),
        cache_dir=preprocess_config.get('cache_dir', 'cache/processed_images'),
        parallel_workers=preprocess_config.get('parallel_workers', 4),
        enabled=preprocess_config.get('enabled', True)
    )
    
    return ImagePreprocessor(config)
