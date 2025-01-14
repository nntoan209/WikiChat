sudo docker run -ti -p 8080:8080 \
    us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cpu.1-4 \
    --model-id BAAI/bge-reranker-v2-m3 \
    --max-client-batch-size 4096 \
    --max-batch-tokens 50000 \
    --max-concurrent-requests 128 \