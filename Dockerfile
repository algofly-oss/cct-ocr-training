FROM python:3.10-slim
RUN apt-get update && apt-get install --no-install-recommends -y build-essential libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt constraints-docker.txt ./
RUN pip install --no-cache-dir -c constraints-docker.txt -r requirements.txt \
    'tensorflow[and-cuda]==2.19.0' 'keras==3.12.1' 'jupyterlab>=4,<5'
ENV PYTHONUNBUFFERED=1
# The repository is bind-mounted by Compose; no datasets or private weights enter the image.
WORKDIR /workspace/notebooks
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8780", "--allow-root", "--no-browser"]
