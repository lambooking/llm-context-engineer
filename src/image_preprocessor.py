"""
增强的图像预处理模块
集成了TIF图像切片处理和原有的预处理功能
"""

import os
import time
import csv
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, field

import rasterio
import numpy as np
from PIL import Image, ImageOps, ImageFile
from loguru import logger
from pathlib import Path

# 配置PIL处理大图像
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class PreprocessConfig:
    """预处理配置"""
    # 原有配置
    max_resolution: Tuple[int, int] = (2048, 2048)
    quality: int = 85
    format: str = "JPEG"
    cache_dir: str = "cache/processed_images"
    parallel_workers: int = 4
    enabled: bool = True

    # TIF切片配置
    tif_processing_enabled: bool = True
    target_tile_size: int = 2048
    grid_rows: int = 3
    grid_cols: int = 3
    tif_jpeg_quality: int = 95
    tif_output_dir: str = "cache/tif_tiles"
    tif_json_dir: str = "cache/tif_json"
    tif_log_csv: str = "cache/tif_processing_log.csv"

    def __post_init__(self):
        """后处理初始化"""
        # 确保TIF相关目录设置正确
        if self.tif_processing_enabled:
            self.tif_output_dir = str(Path(self.cache_dir).parent / "tif_tiles")
            self.tif_json_dir = str(Path(self.cache_dir).parent / "tif_json")
            self.tif_log_csv = str(Path(self.cache_dir).parent / "tif_processing_log.csv")


@dataclass
class TifProcessResult:
    """TIF处理结果"""
    orig_file: str
    resized_image_path: str
    tiles: List[Dict[str, Any]]
    processing_time: float
    json_path: str


class TifProcessor:
    """TIF图像处理器"""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.output_dir = Path(config.tif_output_dir)
        self.json_dir = Path(config.tif_json_dir)
        self.log_csv = config.tif_log_csv

        # 创建必要目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

        # 线程锁
        self._csv_lock = threading.Lock()

        logger.info(f"TIF处理器初始化完成，输出目录: {self.output_dir}")

    def _save_tile(self, img: Image.Image, left: int, top: int,
                   tile_size: int, out_path: str, quality: int = 95) -> Tuple[float, int]:
        """保存单个tile"""
        tile = img.crop((left, top, left + tile_size, top + tile_size))
        t0 = time.time()
        tile.save(out_path, "JPEG", quality=quality)
        t1 = time.time()
        file_size_kb = os.path.getsize(out_path) // 1024
        return round(t1 - t0, 4), file_size_kb

    def process_tif(self, tif_path: str) -> TifProcessResult:
        """
        处理单个TIF文件（优化版本）

        Args:
            tif_path: TIF文件路径

        Returns:
            TifProcessResult: 处理结果
        """
        tif_path = Path(tif_path)
        base_name = tif_path.stem

        logger.info(f"开始处理TIF文件: {tif_path}")
        t_start_total = time.time()

        # 检查文件大小限制
        file_size_mb = os.path.getsize(tif_path) / (1024 * 1024)
        max_size_mb = getattr(self.config, 'max_tif_size_mb', 100)
        max_processing_time = getattr(self.config, 'max_tif_processing_time', 60)
        
        if hasattr(self.config, 'skip_large_tif') and self.config.skip_large_tif and file_size_mb > max_size_mb:
            logger.warning(f"跳过过大的TIF文件: {tif_path} ({file_size_mb:.1f}MB > {max_size_mb}MB)")
            # 返回简化的处理结果
            return self._create_simplified_tif_result(tif_path, base_name)

        # 转换和resize（使用更快的算法）
        t_start_convert = time.time()
        img = self._convert_tif_to_pil_fast(tif_path)

        target_w = self.config.grid_cols * self.config.target_tile_size
        target_h = self.config.grid_rows * self.config.target_tile_size
        
        # 使用更快的resize算法
        img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # 保存resize后的整图
        resized_jpg_path = self.output_dir / f"{base_name}_resized.jpg"
        img.save(resized_jpg_path, "JPEG", quality=self.config.tif_jpeg_quality)
        t_end_convert = time.time()
        convert_time_s = round(t_end_convert - t_start_convert, 4)

        logger.debug(f"TIF转换完成，用时 {convert_time_s}s，目标大小 {target_w}x{target_h}")

        # 切分tiles
        tiles_info = []
        csv_results = []

        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            tasks = []
            for r in range(self.config.grid_rows):
                for c in range(self.config.grid_cols):
                    left = c * self.config.target_tile_size
                    top = r * self.config.target_tile_size
                    out_name = f"{base_name}_col{c}_row{r}_x{left}_y{top}.jpg"
                    out_path = self.output_dir / out_name

                    task = executor.submit(
                        self._save_tile, img, left, top,
                        self.config.target_tile_size, str(out_path),
                        self.config.tif_jpeg_quality
                    )
                    tasks.append((task, r, c, out_name, left, top))

            # 收集结果
            for task, r, c, out_name, left, top in tasks:
                tile_time_s, file_size_kb = task.result()

                tile_info = {
                    "tile_file": out_name,
                    "left": left,
                    "top": top,
                    "width": self.config.target_tile_size,
                    "height": self.config.target_tile_size
                }
                tiles_info.append(tile_info)

                csv_result = {
                    "orig_file": tif_path.name,
                    "tile_file": out_name,
                    "col": c, "row": r,
                    "left": left, "top": top,
                    "width": self.config.target_tile_size,
                    "height": self.config.target_tile_size,
                    "file_kb": file_size_kb,
                    "convert_time_s": convert_time_s,
                    "tile_time_s": tile_time_s
                }
                csv_results.append(csv_result)

                logger.debug(f"保存tile {out_name} {file_size_kb}KB {tile_time_s}s")

        # 保存JSON信息
        json_info = {
            "orig_file": tif_path.name,
            "resized_image": str(resized_jpg_path),
            "tiles": tiles_info
        }
        json_path = self.json_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_info, f, indent=2, ensure_ascii=False)

        # 记录CSV日志
        self._log_to_csv(csv_results)

        t_end_total = time.time()
        processing_time = round(t_end_total - t_start_total, 4)

        logger.info(f"TIF处理完成: {tif_path} 用时 {processing_time}s")

        return TifProcessResult(
            orig_file=tif_path.name,
            resized_image_path=str(resized_jpg_path),
            tiles=tiles_info,
            processing_time=processing_time,
            json_path=str(json_path)
        )

    def _convert_tif_to_pil(self, tif_path: Path) -> Image.Image:
        """将TIF转换为PIL Image"""
        with rasterio.open(tif_path) as src:
            arr = src.read()
            if arr.ndim == 3:
                arr = np.transpose(arr, (1, 2, 0))
            else:
                arr = arr.squeeze(axis=0)

            if arr.dtype != 'uint8':
                arr = arr.astype('float32')
                arr = arr - arr.min()
                if arr.max() > 0:
                    arr = arr / arr.max() * 255.0
                arr = arr.clip(0, 255).astype('uint8')

            return Image.fromarray(arr).convert("RGB")
    
    def _convert_tif_to_pil_fast(self, tif_path: Path) -> Image.Image:
        """快速将TIF转换为PIL Image（优化版本）"""
        try:
            with rasterio.open(tif_path) as src:
                # 读取第一个波段或RGB波段
                if src.count >= 3:
                    # RGB图像，读取前3个波段
                    arr = src.read([1, 2, 3])
                    arr = np.transpose(arr, (1, 2, 0))
                else:
                    # 单波段图像
                    arr = src.read(1)
                
                # 快速数据类型转换
                if arr.dtype != 'uint8':
                    # 使用更快的归一化方法
                    if arr.dtype in ['uint16', 'int16']:
                        arr = (arr / 256).astype('uint8')
                    else:
                        arr = arr.astype('float32')
                        # 使用percentile进行更稳定的归一化
                        p2, p98 = np.percentile(arr, (2, 98))
                        arr = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255).astype('uint8')
                
                # 确保是RGB格式
                if len(arr.shape) == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
                
                return Image.fromarray(arr, mode='RGB')
        except Exception as e:
            logger.warning(f"快速TIF转换失败，使用标准方法: {e}")
            return self._convert_tif_to_pil(tif_path)
    
    def _create_simplified_tif_result(self, tif_path: Path, base_name: str) -> TifProcessResult:
        """为跳过的大文件创建简化的处理结果"""
        # 创建一个简单的占位符图像
        placeholder_path = self.output_dir / f"{base_name}_placeholder.jpg"
        placeholder_img = Image.new('RGB', (512, 512), color='gray')
        placeholder_img.save(placeholder_path, "JPEG", quality=50)
        
        return TifProcessResult(
            orig_file=str(tif_path),
            resized_image_path=str(placeholder_path),
            tiles=[],  # 空的tiles列表
            processing_time=0.1,
            json_path=""
        )

    def _log_to_csv(self, results: List[Dict[str, Any]]):
        """记录结果到CSV"""
        fieldnames = ["orig_file", "tile_file", "col", "row", "left", "top",
                     "width", "height", "file_kb", "convert_time_s", "tile_time_s"]

        with self._csv_lock:
            # 检查文件是否存在，决定是否写入头部
            file_exists = os.path.exists(self.log_csv)

            with open(self.log_csv, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

                for result in results:
                    writer.writerow(result)


class EnhancedImagePreprocessor:
    """增强的图像预处理器"""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化TIF处理器
        if config.tif_processing_enabled:
            self.tif_processor = TifProcessor(config)
        else:
            self.tif_processor = None

        # 线程锁
        self._cache_lock = threading.Lock()

        logger.info(f"增强图像预处理器初始化完成，缓存目录: {self.cache_dir}")

    def preprocess_image(self, image_path: str) -> Union[str, TifProcessResult]:
        """
        预处理单个图像

        Args:
            image_path: 图像路径

        Returns:
            Union[str, TifProcessResult]:
                - 普通图像返回处理后的路径
                - TIF图像返回TifProcessResult对象
        """
        if not self.config.enabled:
            return image_path

        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"图像文件不存在: {image_path}")
            return str(image_path)

        # 检查是否为TIF文件
        if self._is_tif_file(image_path):
            return self._process_tif_image(image_path)
        else:
            return self._process_regular_image(image_path)

    def _is_tif_file(self, image_path: Path) -> bool:
        """检查是否为TIF文件"""
        return image_path.suffix.lower() in ['.tif', '.tiff']

    def _process_tif_image(self, image_path: Path) -> TifProcessResult:
        """处理TIF图像"""
        if not self.config.tif_processing_enabled or not self.tif_processor:
            logger.warning("TIF处理未启用，返回原始路径")
            return str(image_path)

        # 检查缓存
        cache_key = self._generate_tif_cache_key(image_path)
        cached_result = self._get_tif_cache(cache_key)
        if cached_result:
            logger.debug(f"使用TIF缓存: {image_path}")
            return cached_result

        try:
            result = self.tif_processor.process_tif(str(image_path))
            self._save_tif_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"TIF处理失败 {image_path}: {e}")
            return str(image_path)

    def _process_regular_image(self, image_path: Path) -> str:
        """处理常规图像（原有逻辑）"""
        cache_path = self._generate_cache_path(image_path)

        if cache_path.exists():
            logger.debug(f"使用缓存图像: {cache_path}")
            return str(cache_path)

        try:
            # 确保缓存目录存在
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            processed_path = self._process_single_image(image_path, cache_path)
            logger.debug(f"图像预处理完成: {image_path} -> {processed_path}")
            return processed_path
        except Exception as e:
            logger.error(f"图像预处理失败 {image_path}: {e}")
            return str(image_path)

    def preprocess_images_batch(self, image_paths: List[str]) -> List[Union[str, TifProcessResult]]:
        """
        批量预处理图像

        Args:
            image_paths: 图像路径列表

        Returns:
            List[Union[str, TifProcessResult]]: 处理结果列表
        """
        if not self.config.enabled:
            return image_paths

        if not image_paths:
            return []

        logger.info(f"开始批量预处理 {len(image_paths)} 个图像")

        # 分离TIF和普通图像
        tif_paths = []
        regular_paths = []
        path_types = []  # 记录每个路径的类型

        for path in image_paths:
            if self._is_tif_file(Path(path)):
                tif_paths.append(path)
                path_types.append('tif')
            else:
                regular_paths.append(path)
                path_types.append('regular')

        # 处理结果字典
        results = {}

        # 并行处理普通图像
        if regular_paths:
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                future_to_path = {
                    executor.submit(self._process_regular_image, Path(path)): path
                    for path in regular_paths
                }

                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results[path] = result
                    except Exception as e:
                        logger.error(f"批量预处理失败 {path}: {e}")
                        results[path] = path

        # 处理TIF图像（由于可能消耗大量内存，使用较少并发）
        if tif_paths and self.config.tif_processing_enabled:
            tif_workers = min(2, self.config.parallel_workers)  # 限制TIF并发数
            with ThreadPoolExecutor(max_workers=tif_workers) as executor:
                future_to_path = {
                    executor.submit(self._process_tif_image, Path(path)): path
                    for path in tif_paths
                }

                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results[path] = result
                    except Exception as e:
                        logger.error(f"TIF批量处理失败 {path}: {e}")
                        results[path] = path

        # 按原始顺序组装结果
        final_results = []
        for i, path in enumerate(image_paths):
            if path in results:
                final_results.append(results[path])
            else:
                final_results.append(path)

        logger.info("批量预处理完成")
        return final_results

    def _process_single_image(self, input_path: Path, output_path: Path) -> str:
        """处理单个常规图像文件（原有逻辑）"""
        with Image.open(input_path) as img:
            original_size = img.size
            logger.debug(f"原始图像尺寸: {original_size}")

            if self._needs_resize(original_size):
                img = self._resize_image(img)
                logger.debug(f"调整后尺寸: {img.size}")

            if img.mode not in ('RGB', 'L'):
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')

            output_path.parent.mkdir(parents=True, exist_ok=True)

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

        width_ratio = max_width / img.width
        height_ratio = max_height / img.height
        ratio = min(width_ratio, height_ratio)

        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _generate_cache_filename(self, image_path: Path) -> str:
        """生成缓存文件名，保持目录结构"""
        stat = image_path.stat()
        content = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
        config_str = f"{self.config.max_resolution}_{self.config.quality}_{self.config.format}"
        content += f"_{config_str}"

        hash_obj = hashlib.md5(content.encode())
        hash_str = hash_obj.hexdigest()

        if self.config.format.upper() == 'JPEG':
            ext = '.jpg'
        else:
            ext = f'.{self.config.format.lower()}'

        # 保持相对路径结构，避免绝对路径中的特殊字符
        relative_path = image_path
        if image_path.is_absolute():
            # 如果是绝对路径，尝试获取相对于当前工作目录的路径
            try:
                relative_path = image_path.relative_to(Path.cwd())
            except ValueError:
                # 如果无法获取相对路径，使用文件名和部分父目录
                parts = image_path.parts
                if len(parts) >= 2:
                    relative_path = Path(*parts[-2:])  # 取最后两级目录
                else:
                    relative_path = image_path.name
        
        # 将路径转换为安全的缓存路径
        safe_path = str(relative_path).replace('/', '_').replace('\\', '_').replace(':', '_')
        return f"{safe_path}_{hash_str[:8]}{ext}"

    def _generate_cache_path(self, image_path: Path) -> Path:
        """生成缓存文件路径，保持目录结构"""
        stat = image_path.stat()
        content = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
        config_str = f"{self.config.max_resolution}_{self.config.quality}_{self.config.format}"
        content += f"_{config_str}"

        hash_obj = hashlib.md5(content.encode())
        hash_str = hash_obj.hexdigest()[:8]  # 使用较短的哈希

        if self.config.format.upper() == 'JPEG':
            ext = '.jpg'
        else:
            ext = f'.{self.config.format.lower()}'

        # 获取相对路径
        relative_path = image_path
        if image_path.is_absolute():
            try:
                relative_path = image_path.relative_to(Path.cwd())
            except ValueError:
                # 如果无法获取相对路径，使用文件名和部分父目录
                parts = image_path.parts
                if len(parts) >= 3:
                    relative_path = Path(*parts[-3:])  # 取最后三级目录
                elif len(parts) >= 2:
                    relative_path = Path(*parts[-2:])  # 取最后两级目录
                else:
                    relative_path = Path(image_path.name)
        
        # 在缓存目录中保持相同的目录结构
        cache_subdir = self.cache_dir / relative_path.parent
        filename_with_hash = f"{relative_path.stem}_{hash_str}{ext}"
        
        return cache_subdir / filename_with_hash

    def _generate_tif_cache_key(self, tif_path: Path) -> str:
        """生成TIF缓存键"""
        stat = tif_path.stat()
        content = f"{tif_path}_{stat.st_mtime}_{stat.st_size}"
        config_str = f"{self.config.target_tile_size}_{self.config.grid_rows}_{self.config.grid_cols}_{self.config.tif_jpeg_quality}"
        content += f"_{config_str}"

        hash_obj = hashlib.md5(content.encode())
        return hash_obj.hexdigest()

    def _get_tif_cache(self, cache_key: str) -> Optional[TifProcessResult]:
        """获取TIF缓存"""
        cache_file = self.cache_dir / f"tif_cache_{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return TifProcessResult(**data)
            except Exception as e:
                logger.warning(f"读取TIF缓存失败: {e}")
        return None

    def _save_tif_cache(self, cache_key: str, result: TifProcessResult):
        """保存TIF缓存"""
        cache_file = self.cache_dir / f"tif_cache_{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result.__dict__, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存TIF缓存失败: {e}")

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

        # 分类统计
        regular_cache = len([f for f in cache_files if not f.name.startswith('tif_cache_')])
        tif_cache = len([f for f in cache_files if f.name.startswith('tif_cache_')])

        return {
            'cache_dir': str(self.cache_dir),
            'exists': True,
            'file_count': len(cache_files),
            'regular_cache_count': regular_cache,
            'tif_cache_count': tif_cache,
            'total_size': total_size,
            'total_size_mb': round(total_size / 1024 / 1024, 2)
        }

    def clear_cache(self, older_than_days: Optional[int] = None, cache_type: str = 'all'):
        """
        清理缓存

        Args:
            older_than_days: 清理多少天前的缓存，None表示清理所有
            cache_type: 缓存类型 ('all', 'regular', 'tif')
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
                should_delete = False

                if cache_type == 'all':
                    should_delete = True
                elif cache_type == 'regular' and not cache_file.name.startswith('tif_cache_'):
                    should_delete = True
                elif cache_type == 'tif' and cache_file.name.startswith('tif_cache_'):
                    should_delete = True

                if should_delete and cache_file.stat().st_mtime < cutoff_time:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    removed_count += 1
                    removed_size += file_size

        logger.info(f"清理缓存完成: 删除 {removed_count} 个文件，释放 {removed_size/1024/1024:.2f} MB")


class EnhancedPreprocessingPipeline:
    """增强的预处理流水线"""

    def __init__(self, preprocessor: EnhancedImagePreprocessor):
        self.preprocessor = preprocessor
        self._processing_queue = []
        self._processed_cache = {}
        self._tif_results_cache = {}  # 专门缓存TIF处理结果

    def add_to_pipeline(self, image_paths: List[str]) -> List[Any]:
        """
        添加图像到预处理流水线

        Args:
            image_paths: 图像路径列表

        Returns:
            List: 预处理后的结果列表
        """
        processed_results = []
        to_process = []

        for path in image_paths:
            if path in self._processed_cache:
                processed_results.append(self._processed_cache[path])
            else:
                to_process.append(path)
                processed_results.append(None)  # 占位符

        if to_process:
            self._start_async_processing(to_process)

        return processed_results

    def _start_async_processing(self, image_paths: List[str]):
        """启动异步预处理"""
        def process_async():
            processed = self.preprocessor.preprocess_images_batch(image_paths)
            for original, processed_result in zip(image_paths, processed):
                self._processed_cache[original] = processed_result
                # 如果是TIF结果，额外缓存
                if isinstance(processed_result, TifProcessResult):
                    self._tif_results_cache[original] = processed_result

        thread = threading.Thread(target=process_async)
        thread.daemon = True
        thread.start()

    def get_processed_result(self, original_path: str, timeout: float = 30.0) -> Any:
        """
        获取预处理结果，等待处理完成

        Args:
            original_path: 原始路径
            timeout: 超时时间

        Returns:
            处理结果
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if original_path in self._processed_cache:
                return self._processed_cache[original_path]
            time.sleep(0.1)

        logger.warning(f"预处理超时，使用原始路径: {original_path}")
        return original_path

    def get_tif_result(self, original_path: str) -> Optional[TifProcessResult]:
        """获取TIF处理结果"""
        return self._tif_results_cache.get(original_path)


def create_enhanced_preprocessor_from_config(config_dict: Dict[str, Any]) -> EnhancedImagePreprocessor:
    """从配置字典创建增强预处理器"""
    preprocess_config = config_dict.get('image_preprocessing', {})

    config = PreprocessConfig(
        # 原有配置
        max_resolution=tuple(preprocess_config.get('max_resolution', [2048, 2048])),
        quality=preprocess_config.get('quality', 85),
        format=preprocess_config.get('format', 'JPEG'),
        cache_dir=preprocess_config.get('cache_dir', 'cache/processed_images'),
        parallel_workers=preprocess_config.get('parallel_workers', 4),
        enabled=preprocess_config.get('enabled', True),

        # TIF配置
        tif_processing_enabled=preprocess_config.get('tif_processing_enabled', True),
        target_tile_size=preprocess_config.get('target_tile_size', 2048),
        grid_rows=preprocess_config.get('grid_rows', 3),
        grid_cols=preprocess_config.get('grid_cols', 3),
        tif_jpeg_quality=preprocess_config.get('tif_jpeg_quality', 95)
    )

    return EnhancedImagePreprocessor(config)



def get_image_paths_from_folder(folder_path: str, exts: list = None) -> list:
    """
    从文件夹中获取所有图像文件路径

    Args:
        folder_path: 文件夹路径
        exts: 支持的文件扩展名列表，例如 ['.jpg', '.png', '.tif']

    Returns:
        List[str]: 图像文件路径列表
    """
    if exts is None:
        exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"文件夹不存在: {folder_path}")

    image_paths = [str(f) for f in folder.glob("*") if f.suffix.lower() in exts]
    return image_paths


# 使用示例
if __name__ == "__main__":
    # 配置示例
    config_dict = {
        'image_preprocessing': {
            'enabled': True,
            'max_resolution': [2048, 2048],
            'quality': 85,
            'parallel_workers': 4,
            'tif_processing_enabled': True,
            'target_tile_size': 2048,
            'grid_rows': 3,
            'grid_cols': 3,
            'tif_jpeg_quality': 95
        }
    }

    # 创建预处理器
    preprocessor = create_enhanced_preprocessor_from_config(config_dict)

    # 处理图像
    folder_path = "E:/workforzhangjiang/competition-VQA/Gradio/dataFORtest/tif"  # 替换成你的文件夹路径
    image_paths = get_image_paths_from_folder(folder_path)

    # image_paths = ["image1.jpg", "satellite.tif", "image2.png"]
    results = preprocessor.preprocess_images_batch(image_paths)

    for path, result in zip(image_paths, results):
        if isinstance(result, TifProcessResult):
            print(f"TIF处理完成: {path}")
            print(f"  - 切片数量: {len(result.tiles)}")
            print(f"  - 处理时间: {result.processing_time}s")
            print(f"  - JSON路径: {result.json_path}")
        else:
            print(f"常规图像处理完成: {path} -> {result}")