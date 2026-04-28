FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter kernel
RUN python -m ipykernel install --sys-prefix --name cse151b --display-name "Python (cse151b)"

# Copy project files
COPY . .

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
