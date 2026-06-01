# CSE 151B SP26 Kaggle Competition

**Team:** AAMJ — Adam Connor, Amadeus Karna, Madhav Baghla, Jesse Huang

**GPU & inference time:** NVIDIA T4 (15 GB VRAM). Approximate total inference time: 6-8 hours on the full private test set.

**Model weights:** Downloaded automatically from HuggingFace on first run. Accept the model license at https://huggingface.co/Qwen/Qwen3-4B and set your `HF_TOKEN` environment variable (or Colab/Kaggle secret) before running.

**Reproducing results:** Call `run_inference()` from Python:
```python
from run_inference import run_inference
run_inference()
```
Or run directly from the command line: `python run_inference.py`
