export CUDA_VISIBLE_DEVICES=4
echo "Current CUDA devices: $CUDA_VISIBLE_DEVICES"
python -m vllm.entrypoints.openai.api_server --model /workspace/models/Qwen/Qwen2___5-VL-7B-Instruct --served-model-name qwen-vl --port 1238 --gpu-memory-utilization 0.8 --tensor-parallel-size 1