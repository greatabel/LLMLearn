# LLMLearn


# --- 访问server端jupyter ---
ubuntu server:
jupyter notebook  gpt-dev.ipynb  --no-browser --port=8888 --ip=0.0.0.0

mac client:
ssh -N -f -L localhost:8888:localhost:8888 3090

然后访问 ubuntu server 出来的：
http://127.0.0.1:8888/tree?token=48481de524dcac6af6b64c4923fd4de7b543b8aee3248a47
类似地址
然后访问：
http://127.0.0.1:8888/notebooks/gpt-dev.ipynb