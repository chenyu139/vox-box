import os
import re
import sys
import wave
import numpy as np
import tempfile
import torch
from typing import Dict, List, Optional, Generator

from vox_box.backends.tts.base import TTSBackend
from vox_box.utils.log import log_method
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.utils.audio import convert
from vox_box.utils.model import create_model_dict

paths_to_insert = [
    os.path.join(os.path.dirname(__file__), "../../third_party/CosyVoice"),
    os.path.join(
        os.path.dirname(__file__), "../../third_party/CosyVoice/third_party/Matcha-TTS"
    ),
]

builtin_spk2info_path = os.path.join(os.path.dirname(__file__), "cosyvoice_spk2info.pt")


class CosyVoice(TTSBackend):
    def __init__(
        self,
        cfg: Config,
    ):
        self.model_load = False
        self.language_map = {
            "中文女": "Chinese Female",
            "中文男": "Chinese Male",
            "日语男": "Japanese Male",
            "粤语女": "Cantonese Female",
            "英文女": "English Female",
            "英文男": "English Male",
            "韩语女": "Korean Female",
        }
        self.reverse_language_map = {v: k for k, v in self.language_map.items()}
        self._cfg = cfg
        self._voices = None
        self._model = None
        self._model_dict = {}
        self._is_cosyvoice_v2 = False

        self._parse_and_set_cuda_visible_devices()

        # 检查两种可能的配置文件
        cosyvoice_yaml_path = os.path.join(self._cfg.model, "cosyvoice.yaml")
        cosyvoice2_yaml_path = os.path.join(self._cfg.model, "cosyvoice2.yaml")

        # 优先检查cosyvoice2.yaml，如果不存在则检查cosyvoice.yaml
        config_path = None
        if os.path.exists(cosyvoice2_yaml_path):
            config_path = cosyvoice2_yaml_path
        elif os.path.exists(cosyvoice_yaml_path):
            config_path = cosyvoice_yaml_path

        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
                if re.search(r"Qwen2", content, re.IGNORECASE):
                    self._is_cosyvoice_v2 = True

    def _parse_and_set_cuda_visible_devices(self):
        """
        Parse CUDA device in format cuda:1 and set CUDA_VISIBLE_DEVICES accordingly.
        """
        device = self._cfg.device
        if device.startswith("cuda:"):
            device_index = device.split(":")[1]
            if device_index.isdigit():
                os.environ["CUDA_VISIBLE_DEVICES"] = device_index
            else:
                raise ValueError(f"Invalid CUDA device index: {device_index}")

    def load(self):
        for path in paths_to_insert:
            sys.path.insert(0, path)

        if self.model_load:
            return self

        if self._is_cosyvoice_v2:
            from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoiceModel2

            self._model = CosyVoiceModel2(self._cfg.model)

            # CosyVoice2 does not have builtin spk2info.pt
            if not self._model.frontend.spk2info:
                self._model.frontend.spk2info = torch.load(builtin_spk2info_path)
        else:
            from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceModel

            self._model = CosyVoiceModel(self._cfg.model)

        self._voices = self._get_voices()
        self._model_dict = create_model_dict(
            self._cfg.model,
            task_type=TaskTypeEnum.TTS,
            backend_framework=BackendEnum.COSY_VOICE,
            voices=self._voices,
        )

        self.model_load = True
        return self

    def is_load(self) -> bool:
        return self.model_load

    def model_info(self) -> Dict:
        return self._model_dict

    @log_method
    def speech(
        self,
        input: str,
        voice: Optional[str] = "Chinese Female",
        speed: float = 1,
        reponse_format: str = "wav",
        **kwargs,
    ) -> str:
        if voice not in self._voices:
            raise ValueError(f"Voice {voice} not supported")

        original_voice = self._get_original_voice(voice)
        model_output = self._model.inference_sft(
            input, original_voice, stream=False, speed=speed
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            wav_file_path = temp_file.name
            with wave.open(wav_file_path, "wb") as wf:
                wf.setnchannels(1)  # single track
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(22050)  # Sample rate
                for i in model_output:
                    tts_audio = (
                        (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
                    )
                    wf.writeframes(tts_audio)

                output_file_path = convert(wav_file_path, reponse_format, speed)
                return output_file_path

    @log_method
    def speech_instruct(
        self,
        input: str,
        instruct_text: str,
        speech,
        speed: float = 1,
        response_format: str = "wav",
        **kwargs,
    ) -> str:
        # 调用CosyVoice的inference_instruct2方法
        model_output = self._model.inference_instruct2(input, instruct_text, speech)

        # 生成输出音频文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            wav_file_path = temp_file.name
            with wave.open(wav_file_path, "wb") as wf:
                wf.setnchannels(1)  # single track
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(22050)  # Sample rate
                for i in model_output:
                    tts_audio = (
                        (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
                    )
                    wf.writeframes(tts_audio)

            output_file_path = convert(wav_file_path, response_format, speed)
            return output_file_path

    @log_method
    def speech_zero_shot(
        self,
        input: str,
        prompt_text: str,
        speech,
        speed: float = 1,
        response_format: str = "wav",
        **kwargs,
    ) -> str:
        # 调用CosyVoice的inference_zero_shot方法
        model_output = self._model.inference_zero_shot(input, prompt_text, speech)

        # 生成输出音频文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            wav_file_path = temp_file.name
            with wave.open(wav_file_path, "wb") as wf:
                wf.setnchannels(1)  # single track
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(22050)  # Sample rate
                for i in model_output:
                    tts_audio = (
                        (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
                    )
                    wf.writeframes(tts_audio)

            output_file_path = convert(wav_file_path, response_format, speed)
            return output_file_path

    @log_method
    def speech_zero_shot_stream(
        self,
        input: str,
        prompt_text: str,
        speech,
        speed: float = 1,
        response_format: str = "wav",
        **kwargs,
    ) -> Generator[bytes, None, None]:
        """流式输出零样本语音合成音频数据"""
        # 调用CosyVoice的inference_zero_shot方法，利用其流式输出特性
        model_output = self._model.inference_zero_shot(
            input, prompt_text, speech, stream=True
        )

        # 音频参数配置
        sample_rate = 22050
        channels = 1
        sample_width = 2  # 16-bit
        chunk_counter = 0  # 添加计数器用于文件编号

        def create_wav_chunk(audio_data):
            """创建WAV格式的音频块，包含完整的WAV头"""
            nonlocal chunk_counter
            chunk_counter += 1

            # 使用固定文件名保存临时文件
            wav_file_path = f"{chunk_counter}.wav"
            print(f"保存音频文件: {os.path.abspath(wav_file_path)}")  # 打印文件绝对路径

            with wave.open(wav_file_path, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

            # 转换格式
            if response_format != "wav":
                output_file_path = convert(wav_file_path, response_format, speed)
                print(
                    f"转换后文件: {os.path.abspath(output_file_path)}"
                )  # 打印转换后文件路径
                with open(output_file_path, "rb") as f:
                    chunk_data = f.read()
                # 注意：这里不删除文件，保留用于测试
                # if os.path.exists(output_file_path):
                #     os.unlink(output_file_path)
                return chunk_data
            else:
                with open(wav_file_path, "rb") as f:
                    return f.read()
                # 注意：这里不删除wav文件，保留用于测试

        # 利用模型的流式输出，立即处理每个音频片段
        for audio_chunk in model_output:
            tts_audio_bytes = (
                (audio_chunk["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
            )
            # 立即输出每个音频块
            yield create_wav_chunk(tts_audio_bytes)

    @log_method
    def speech_instruct_stream(
        self,
        input: str,
        instruct_text: str,
        speech,
        speed: float = 1,
        response_format: str = "wav",
        **kwargs,
    ) -> Generator[bytes, None, None]:
        """流式输出音频数据的测试方法，利用inference_instruct2的流式特性"""
        # 调用CosyVoice的inference_instruct2方法，利用其流式输出特性
        model_output = self._model.inference_instruct2(input, instruct_text, speech)

        # 音频参数配置
        sample_rate = 22050
        channels = 1
        sample_width = 2  # 16-bit
        chunk_counter = 0  # 添加计数器用于文件编号

        def create_wav_chunk(audio_data):
            """创建WAV格式的音频块，包含完整的WAV头"""
            nonlocal chunk_counter
            chunk_counter += 1

            # 使用固定文件名保存临时文件
            wav_file_path = f"{chunk_counter}.wav"
            print(f"保存音频文件: {os.path.abspath(wav_file_path)}")  # 打印文件绝对路径

            with wave.open(wav_file_path, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

            # 转换格式
            if response_format != "wav":
                output_file_path = convert(wav_file_path, response_format, speed)
                print(
                    f"转换后文件: {os.path.abspath(output_file_path)}"
                )  # 打印转换后文件路径
                with open(output_file_path, "rb") as f:
                    chunk_data = f.read()
                # 注意：这里不删除文件，保留用于测试
                # if os.path.exists(output_file_path):
                #     os.unlink(output_file_path)
                return chunk_data
            else:
                with open(wav_file_path, "rb") as f:
                    return f.read()
                # 注意：这里不删除wav文件，保留用于测试

        # 利用模型的流式输出，立即处理每个音频片段
        for audio_chunk in model_output:
            tts_audio_bytes = (
                (audio_chunk["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
            )
            # 立即输出每个音频块
            yield create_wav_chunk(tts_audio_bytes)

    def _get_voices(self) -> List[str]:
        voices = self._model.list_available_spks()
        return [self.language_map.get(voice, voice) for voice in voices]

    def _get_original_voice(self, voice: str) -> str:
        return self.reverse_language_map.get(voice, voice)
