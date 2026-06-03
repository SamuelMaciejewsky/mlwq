#!/usr/bin/bash
python run.py meta-llama/llama-3.2-3B wikitext2 3bit --device "cuda:0" --tasks piqa
