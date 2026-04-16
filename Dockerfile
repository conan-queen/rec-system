# ── 构建阶段 ──────────────────────────────────────────────────
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11-slim
# 设置工作目录
WORKDIR /app

# 先复制依赖文件（利用 Docker 层缓存，依赖没变就不重新安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 训练模型（构建镜像时就完成训练，启动容器时直接加载）
EXPOSE 8000

CMD ["sh", "-c", "python train.py && uvicorn api.server:app --host 0.0.0.0 --port 8000"]
