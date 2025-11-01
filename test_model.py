from transformers import AutoTokenizer

try:
    AutoTokenizer.from_pretrained('google/embeddinggemma-300m')
    print('Model found')
except Exception as e:
    print(e)