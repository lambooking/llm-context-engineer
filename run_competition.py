#!/usr/bin/env python3
"""
比赛用统一启动脚本 - 启动vLLM服务并运行批量处理器
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading
import yaml
from pathlib import Path
from typing import Optional
import requests
import logging

# 设置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompetitionRunner:
    def __init__(self, config_path: str):
        """初始化比赛运行器"""
        self.config_path = config_path
        self.config = self.load_config()
        self.vllm_process = None
        self.shutdown_event = threading.Event()
        
    def load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"已加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    def start_vllm_server(self) -> bool:
        """启动vLLM服务器"""
        vllm_config = self.config.get('vllm_server', {})
        
        # 设置CUDA设备
        cuda_devices = vllm_config.get('cuda_visible_devices', '4')
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        logger.info(f"设置CUDA设备: {cuda_devices}")
        
        # 构建vLLM启动命令
        cmd = [
            'python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', vllm_config.get('model_path', '/workspace/models/Qwen_Qwen2.5-VL-7B-Instruct/Qwen/Qwen2___5-VL-7B-Instruct'),
            '--served-model-name', vllm_config.get('served_model_name', 'qwen-vl'),
            '--port', str(vllm_config.get('port', 1238)),
            '--gpu-memory-utilization', str(vllm_config.get('gpu_memory_utilization', 0.8)),
            '--tensor-parallel-size', str(vllm_config.get('tensor_parallel_size', 1))
        ]
        
        logger.info(f"启动vLLM服务器: {' '.join(cmd)}")
        
        try:
            # 启动vLLM服务器进程
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 启动日志输出线程
            log_thread = threading.Thread(target=self._log_vllm_output)
            log_thread.daemon = True
            log_thread.start()
            
            # 等待服务器启动
            if self.wait_for_server():
                logger.info("vLLM服务器启动成功")
                return True
            else:
                logger.error("vLLM服务器启动失败")
                return False
                
        except Exception as e:
            logger.error(f"启动vLLM服务器时发生错误: {e}")
            return False
    
    def _log_vllm_output(self):
        """记录vLLM服务器输出"""
        if self.vllm_process:
            for line in iter(self.vllm_process.stdout.readline, ''):
                if self.shutdown_event.is_set():
                    break
                logger.info(f"[vLLM] {line.strip()}")
    
    def wait_for_server(self, timeout: int = 300) -> bool:
        """等待服务器启动完成"""
        port = self.config.get('vllm_server', {}).get('port', 1238)
        url = f"http://localhost:{port}/v1/models"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False
                
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("服务器健康检查通过")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(5)
            logger.info("等待服务器启动...")
        
        return False
    
    def run_batch_processor(self, input_path: str, output_path: str) -> int:
        """运行批量处理器"""
        cmd = [
            'python', 'batch_processor.py',
            input_path,
            output_path,
            '--config', self.config_path,
            '--log-level', 'INFO'
        ]
        
        logger.info(f"启动批量处理器: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True)
            logger.info("批量处理器执行完成")
            return result.returncode
        except subprocess.CalledProcessError as e:
            logger.error(f"批量处理器执行失败: {e}")
            return e.returncode
        except Exception as e:
            logger.error(f"运行批量处理器时发生错误: {e}")
            return 1
    
    def shutdown(self):
        """关闭所有服务"""
        logger.info("开始关闭服务...")
        self.shutdown_event.set()
        
        if self.vllm_process:
            logger.info("正在关闭vLLM服务器...")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM服务器未能正常关闭，强制终止")
                self.vllm_process.kill()
                self.vllm_process.wait()
            logger.info("vLLM服务器已关闭")
    
    def run(self, input_path: str, output_path: str) -> int:
        """运行完整流程"""
        try:
            # 启动vLLM服务器
            if not self.start_vllm_server():
                logger.error("无法启动vLLM服务器")
                return 1
            
            # 运行批量处理器
            return_code = self.run_batch_processor(input_path, output_path)
            
            return return_code
            
        finally:
            self.shutdown()


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info("收到终止信号，正在关闭...")
    if hasattr(signal_handler, 'runner'):
        signal_handler.runner.shutdown()
    sys.exit(0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='比赛用统一启动脚本')
    parser.add_argument('input_path', help='输入文件路径')
    parser.add_argument('output_path', help='输出文件路径')
    parser.add_argument('--config', default='config/config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 验证输入参数
    if not os.path.exists(args.input_path):
        logger.error(f"输入路径不存在: {args.input_path}")
        return 1
    
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        return 1
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 设置信号处理器
    runner = CompetitionRunner(args.config)
    signal_handler.runner = runner
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行
    logger.info("="*50)
    logger.info("启动比赛用统一脚本")
    logger.info(f"输入路径: {args.input_path}")
    logger.info(f"输出路径: {args.output_path}")
    logger.info(f"配置文件: {args.config}")
    logger.info("="*50)
    
    return runner.run(args.input_path, args.output_path)


if __name__ == "__main__":
    sys.exit(main())
