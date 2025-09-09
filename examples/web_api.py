#!/usr/bin/env python3
"""
Web API示例 - 使用Flask创建REST API服务
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import tempfile
import uuid

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import base64

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.main_processor import ProcessorFactory
from src.constants import QuestionType

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局处理器实例
processor = None

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_processor():
    """初始化处理器"""
    global processor
    if processor is None:
        processor = ProcessorFactory.create_from_env()
    return processor


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'service': 'LLM Context Engineer API',
        'version': '1.0.0'
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """获取统计信息"""
    try:
        proc = init_processor()
        stats = proc.get_stats()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/classify', methods=['POST'])
def classify_question():
    """问题分类预览端点"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing question parameter'
            }), 400
        
        question = data['question']
        image_count = data.get('image_count', 1)
        
        proc = init_processor()
        result = proc.get_classification_preview(question, image_count)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/query', methods=['POST'])
def process_query():
    """处理查询端点"""
    try:
        # 获取表单数据
        question = request.form.get('question')
        if not question:
            return jsonify({
                'success': False,
                'error': 'Missing question parameter'
            }), 400
        
        # 强制类型（可选）
        force_type = request.form.get('force_type')
        force_type_enum = None
        if force_type:
            try:
                force_type_enum = QuestionType[force_type.upper()]
            except KeyError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid question type: {force_type}'
                }), 400
        
        # 处理上传的图片文件
        image_paths = []
        temp_files = []
        
        try:
            files = request.files.getlist('images')
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    # 创建临时文件
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=f".{file.filename.rsplit('.', 1)[1].lower()}"
                    )
                    file.save(temp_file.name)
                    temp_files.append(temp_file.name)
                    image_paths.append(temp_file.name)
            
            # 处理base64编码的图片
            base64_images = request.form.getlist('base64_images')
            for i, b64_data in enumerate(base64_images):
                if b64_data:
                    try:
                        # 解析base64数据
                        if ',' in b64_data:
                            header, data = b64_data.split(',', 1)
                        else:
                            data = b64_data
                        
                        # 解码并保存到临时文件
                        image_data = base64.b64decode(data)
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".jpg"
                        )
                        temp_file.write(image_data)
                        temp_file.close()
                        temp_files.append(temp_file.name)
                        image_paths.append(temp_file.name)
                        
                    except Exception as e:
                        print(f"处理base64图片失败: {e}")
            
            # 初始化处理器并处理查询
            proc = init_processor()
            result = proc.process_query(
                question=question,
                images=image_paths,
                force_type=force_type_enum
            )
            
            # 清理敏感信息
            if 'raw_response' in result:
                del result['raw_response']
            
            return jsonify(result)
            
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch', methods=['POST'])
def batch_process():
    """批量处理端点"""
    try:
        data = request.get_json()
        if not data or 'queries' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing queries parameter'
            }), 400
        
        queries = data['queries']
        if not isinstance(queries, list):
            return jsonify({
                'success': False,
                'error': 'queries must be a list'
            }), 400
        
        proc = init_processor()
        results = proc.batch_process(queries)
        
        # 清理敏感信息
        for result in results:
            if 'raw_response' in result:
                del result['raw_response']
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/prompts', methods=['GET'])
def get_prompts():
    """获取所有Prompt模板"""
    try:
        proc = init_processor()
        prompts = {}
        
        for question_type in QuestionType:
            prompt = proc.prompt_manager.get_prompt(question_type, question="示例问题")
            prompts[question_type.name] = {
                'name': question_type.value,
                'template': prompt
            }
        
        return jsonify({
            'success': True,
            'data': prompts
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/prompts/<question_type>', methods=['PUT'])
def update_prompt(question_type):
    """更新指定类型的Prompt模板"""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing prompt parameter'
            }), 400
        
        # 验证问题类型
        try:
            qtype = QuestionType[question_type.upper()]
        except KeyError:
            return jsonify({
                'success': False,
                'error': f'Invalid question type: {question_type}'
            }), 400
        
        proc = init_processor()
        proc.update_prompt(qtype, data['prompt'])
        
        return jsonify({
            'success': True,
            'message': f'Updated prompt for {qtype.value}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large'
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # 设置环境变量
    os.environ.setdefault('VLM_API_KEY', 'your_api_key_here')
    os.environ.setdefault('VLM_PROVIDER', 'openai')
    
    print("启动LLM Context Engineer API服务...")
    print("API文档:")
    print("  GET  /health          - 健康检查")
    print("  GET  /stats           - 获取统计信息")
    print("  POST /classify        - 问题分类预览")
    print("  POST /query           - 处理单个查询")
    print("  POST /batch           - 批量处理")
    print("  GET  /prompts         - 获取所有Prompt")
    print("  PUT  /prompts/<type>  - 更新Prompt")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
