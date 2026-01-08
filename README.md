install dependencies:
1. Clone the repo:
```
git clone apurba_tts
```

2. install uv if you haven't already
cd apurba_tts && pip install uv (if you do)

3. sync:
```
uv sync
```

To run:
```
uv run main.py --model_dir Spark-TTS-0.5B --device 0 --host 0.0.0.0 --port 8501
```