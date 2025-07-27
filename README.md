# **MLWQ: Efficient Small Language Model Deployment via Multi-Level Weight Quantization**

MLWQ facilitates a more effective bit-width allocation strategy by jointly optimizing inter-layer loss and intra-layer salience. Moreover, we introduce a fine-grained partitioning of intra-layer salience, which enables precise adjustment of quantization parameters within each salience group.

---

- **Requirements**
```bash
Miniconda (recommended)
NVIDIA GPU
```
___
- **Install**

```bash
conda create -n mlwqllm python=3.10 -y  
conda activate mlwqllm  
git clone https://github.com/hudevictor/mlwq.git
cd mlwq
pip install --upgrade pip 
pip install -r requirements.txt
```
___
- **Usage**
```bash
(1) PPL
cd mlwq
cd scripts
./llama.sh
(2) Zero-shot
cd mlwq
python run.py  llama-3.2-3B wikitext2 3bit --device "cuda:0" --tasks piqa
```
