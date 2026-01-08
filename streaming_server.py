# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
import logging
import argparse
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional
import tempfile
import base64
import unsloth

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from cli.SparkTTS import SparkTTS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SparkTTS Streaming Server")

from huggingface_hub import snapshot_download, login

def download_model(repo_id="SibgatUl/spark_tts_bn", local_dir="./pretrained_models/"):

    # login(token=os.getenv("HF_TOKEN"))
    
    get_model = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir + repo_id.split("/")[-1],
        cache_dir="./model_cache/",
        token=os.getenv("HF_TOKEN")
    )

    return get_model


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[SparkTTS] = None


def initialize_model(model_dir: str, device: int = 0):
    """Initialize the SparkTTS model."""

    # lora_dir = download_model("SibgatUl/spark_tts_kaggle")
    lora_dir = download_model("SibgatUl/spark_tts_bn")
    
    # snapshot_download(
    #     "unsloth/Spark-TTS-0.5B", 
    #     local_dir="./Spark-TTS-0.5B"
    # )

    global model
    
    logger.info(f"Loading model from: {model_dir}")
    
    if torch.cuda.is_available():
        device_obj = torch.device(f"cuda:{device}")
        logger.info(f"Using CUDA device: {device_obj}")
    else:
        device_obj = torch.device("cpu")
        logger.info("Using CPU")
    
    model = SparkTTS(model_dir, lora_dir=lora_dir, load_lora=True, device=device_obj, streaming=False, use_unsloth=True)
    logger.info("Model loaded successfully")


@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the HTML interface."""
    html_path = Path(__file__).parent / "webui_streaming.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>SparkTTS Streaming</title></head>
            <body>
                <h1>SparkTTS Streaming Server</h1>
                <p>Server is running. HTML UI not found at webui_streaming.html</p>
                <p>Connect to WebSocket endpoint: ws://localhost:8000/tts</p>
            </body>
        </html>
        """


@app.websocket("/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS."""
    await websocket.accept()
    logger.info("Client connected")
    
    try:
        # Receive request from client
        data = await websocket.receive_json()
        
        text = data.get("text", "")
        mode = data.get("mode", "creation")  # 'creation' or 'clone'
        
        # Voice creation parameters
        gender = data.get("gender", "female")
        pitch = data.get("pitch", "moderate")
        speed = data.get("speed", "moderate")
        
        # Voice cloning parameters
        prompt_text = data.get("prompt_text", None)
        prompt_audio_base64 = data.get("prompt_audio", None)
        
        # Sampling parameters
        temperature = data.get("temperature", 0.8)
        top_k = data.get("top_k", 50)
        top_p = data.get("top_p", 0.95)
        chunk_size = data.get("chunk_size", 50)
        
        logger.info(f"Received request - Mode: {mode}, Text: {text[:50]}...")
        
        if not text:
            await websocket.send_json({"error": "No text provided"})
            return
        
        # Handle voice cloning mode
        prompt_speech_path = None
        if mode == "clone" and prompt_audio_base64:
            # Decode base64 audio and save to temp file
            try:
                audio_data = base64.b64decode(prompt_audio_base64)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    prompt_speech_path = Path(f.name)
                logger.info(f"Prompt audio saved to: {prompt_speech_path}")
            except Exception as e:
                logger.error(f"Error processing prompt audio: {e}")
                await websocket.send_json({"error": f"Invalid audio data: {str(e)}"})
                return
        
        # Get event loop for running blocking code
        loop = asyncio.get_running_loop()
        
        # Stream audio chunks
        try:
            if mode == "clone":
                # Voice cloning streaming
                generator = model.inference_stream(
                    text=text,
                    prompt_speech_path=prompt_speech_path,
                    prompt_text=prompt_text,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    chunk_size=chunk_size,
                )
            else:
                # Voice creation streaming
                generator = model.inference_stream(
                    text=text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    chunk_size=chunk_size,
                )
            
            chunk_count = 0
            for wav_chunk in generator:
                # Ensure wav_chunk is float32 numpy array
                if isinstance(wav_chunk, torch.Tensor):
                    wav_chunk = wav_chunk.cpu().numpy()
                
                wav_chunk = np.asarray(wav_chunk, dtype=np.float32)
                wav_chunk = np.clip(wav_chunk, -1.0, 1.0)
                
                # Send binary audio data
                await websocket.send_bytes(wav_chunk.tobytes())
                chunk_count += 1
                logger.debug(f"Sent chunk {chunk_count}, size: {len(wav_chunk)}")
            
            # Send end-of-stream marker (empty bytes)
            await websocket.send_bytes(b"")
            logger.info(f"Streaming completed. Total chunks sent: {chunk_count}")
            
        finally:
            # Clean up temp file if created
            if prompt_speech_path and prompt_speech_path.exists():
                try:
                    os.unlink(prompt_speech_path)
                    logger.info("Cleaned up temporary prompt audio file")
                except Exception as e:
                    logger.warning(f"Could not delete temp file: {e}")
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if model else None
    }


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SparkTTS Streaming Server")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host/IP."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize model
    initialize_model(args.model_dir, args.device)
    
    # Run server
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
