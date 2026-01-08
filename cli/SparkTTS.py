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

import re
import torch
from typing import Tuple, Generator
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, TextStreamer
from threading import Thread
import numpy as np

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP

import unsloth
from unsloth import FastModel

class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(self, model_dir: Path, lora_dir: Path = None, load_lora: bool = False, device: torch.device = torch.device("cuda:0"), streaming: bool = False, use_unsloth: bool = False):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = 16000
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.streaming = streaming
        self.lora_dir = lora_dir
        self.load_lora = load_lora

        if use_unsloth:
            from unsloth import FastModel

            self._initialize_inference_UNSLOTH()
        else:
            self._initialize_inference()

        if streaming:
            from transformers import TextStreamer
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=False)


    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.model.to(self.device)

    def _initialize_inference_UNSLOTH(self):
        if self.lora_dir is not None and self.load_lora:
            self.model, self.tokenizer = FastModel.from_pretrained(
                f"{self.lora_dir}",
                max_seq_length = 2048,
                dtype = torch.float32,
                full_finetuning = False,
                load_in_4bit = False,
            )
        else:
            self.model, self.tokenizer = FastModel.from_pretrained(
                f"{self.model_dir}/LLM",
                max_seq_length = 2048,
                dtype = torch.float32,
                full_finetuning = False,
                load_in_4bit = False,
            )

        self.model.to(self.device, torch.bfloat16)
        FastModel.for_inference(self.model) 

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)

        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Generate speech using the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=True,
            top_k=top_k,
            top_p=1,
            temperature=temperature,
            streamer=self.streamer if self.streaming else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Trim the output tokens to remove the input tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the generated tokens into text
        predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )

        if gender is not None:
            global_token_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        # Convert semantic tokens back to waveform
        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device),
        )

        return wav

    @torch.no_grad()
    def inference_stream(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        chunk_size: int = 50,
    ) -> Generator[np.ndarray, None, None]:
        """
        Performs streaming inference to generate speech from text, yielding audio chunks as they're generated.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.
            chunk_size (int, optional): Number of semantic tokens per audio chunk. Default is 50.

        Yields:
            np.ndarray: Generated waveform chunks as numpy arrays.
        """
        if gender is not None:
            if gender == "female":
                prompt_speech_path = Path(f"cli/LJ001-0001.wav")
            else:
                # prompt = self.process_prompt_control(gender, pitch, speed, text)
                prompt_speech_path = Path(f"cli/en_male_1.wav")
            
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )

        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
        
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=False
        )

        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": 1024,
            "do_sample": True,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        token_buffer = ""
        semantic_tokens_list = []
        
        # Pattern to match semantic tokens
        semantic_pattern = re.compile(r"bicodec_semantic_(\d+)")
        global_pattern = re.compile(r"bicodec_global_(\d+)")

        # Process streamed tokens
        for text_chunk in streamer:
            token_buffer += text_chunk
            
            # Extract global tokens if in controllable mode
            if gender is not None and global_token_ids is None:
                global_matches = global_pattern.findall(token_buffer)
                if global_matches:
                    global_token_ids = (
                        torch.tensor([int(token) for token in global_matches])
                        .long()
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
            
            # Extract semantic tokens
            semantic_matches = semantic_pattern.findall(token_buffer)
            if semantic_matches:
                for match in semantic_matches:
                    semantic_tokens_list.append(int(match))
                
                # Clear processed tokens from buffer
                token_buffer = semantic_pattern.sub("", token_buffer, count=len(semantic_matches))
                
                # Generate audio when we have enough tokens
                if len(semantic_tokens_list) >= chunk_size:
                    pred_semantic_ids = torch.tensor(
                        semantic_tokens_list[:chunk_size]
                    ).long().unsqueeze(0)
                    
                    if global_token_ids is not None:
                        wav_chunk = self.audio_tokenizer.detokenize(
                            global_token_ids.to(self.device).squeeze(0),
                            pred_semantic_ids.to(self.device),
                        )
                        yield wav_chunk
                    
                    # Remove processed tokens
                    semantic_tokens_list = semantic_tokens_list[chunk_size:]

        # Wait for generation to complete
        thread.join()

        # Process any remaining tokens
        if semantic_tokens_list and global_token_ids is not None:
            pred_semantic_ids = torch.tensor(
                semantic_tokens_list
            ).long().unsqueeze(0)
            
            wav_chunk = self.audio_tokenizer.detokenize(
                global_token_ids.to(self.device).squeeze(0),
                pred_semantic_ids.to(self.device),
            )
            yield wav_chunk
