import os
import sys
import logging
import tempfile
import uuid
from typing import Dict, List, Optional

from vox_box.backends.tts.base import TTSBackend
from vox_box.utils.log import log_method
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.utils.audio import convert
from vox_box.utils.model import create_model_dict

logger = logging.getLogger(__name__)

# 添加ACE-Step路径到sys.path
paths_to_insert = [
    os.path.join(os.path.dirname(__file__), "../../third_party/ACE-Step"),
]


class AceStep(TTSBackend):
    def __init__(self, cfg: Config):
        self.model_load = False
        self._cfg = cfg
        self._model = None
        self._model_dict = {}

        # 设置CUDA设备
        self._parse_and_set_cuda_visible_devices()

        # 默认参数配置，基于infer.py和infer-api.py
        self.default_params = {
            "bf16": True,
            "torch_compile": False,
            "cpu_offload": False,
            "overlapped_decode": False,
            "audio_duration": 10.0,
            "infer_step": 50,
            "guidance_scale": 7.5,
            "scheduler_type": "ddim",
            "cfg_type": "full",
            "omega_scale": 1.0,
            "guidance_interval": 1.0,
            "guidance_interval_decay": 0.0,
            "min_guidance_scale": 1.0,
            "use_erg_tag": True,
            "use_erg_lyric": True,
            "use_erg_diffusion": True,
            "oss_steps": [0],
            "guidance_scale_text": 0.0,
            "guidance_scale_lyric": 0.0,
            "actual_seeds": [42],
        }

    def _parse_and_set_cuda_visible_devices(self):
        """解析CUDA设备配置"""
        device = self._cfg.device
        if device.startswith("cuda:"):
            device_index = device.split(":")[1]
            if device_index.isdigit():
                os.environ["CUDA_VISIBLE_DEVICES"] = device_index
            else:
                raise ValueError(f"Invalid CUDA device index: {device_index}")

    def load(self):
        """加载AceStep模型"""
        if self.model_load:
            return self

        # 添加ACE-Step路径到sys.path
        for path in paths_to_insert:
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            # 导入ACE-Step相关模块
            from acestep.pipeline_ace_step import ACEStepPipeline

            # 初始化模型
            self._model = ACEStepPipeline(
                checkpoint_dir=self._cfg.model,
                dtype="bfloat16" if self.default_params["bf16"] else "float32",
                torch_compile=self.default_params["torch_compile"],
                cpu_offload=self.default_params["cpu_offload"],
                overlapped_decode=self.default_params["overlapped_decode"],
            )

            # 创建模型信息字典
            self._model_dict = create_model_dict(
                self._cfg.model,
                task_type=TaskTypeEnum.TTS,
                backend_framework=BackendEnum.ACESTEP,
            )

            self.model_load = True
            logger.info("AceStep model loaded successfully")
            return self

        except Exception as e:
            logger.error(f"Failed to load AceStep model: {e}")
            raise

    def is_load(self) -> bool:
        return self.model_load

    def model_info(self) -> Dict:
        return self._model_dict

    @log_method
    def speech(
        self,
        input: str,
        voice: Optional[str] = None,
        speed: float = 1,
        response_format: str = "mp3",
        **kwargs,
    ) -> str:
        """生成背景音乐/音频

        Args:
            input: 文本提示词(prompt)
            voice: 未使用，保持接口兼容性
            speed: 未使用，保持接口兼容性
            response_format: 输出格式
            **kwargs: 其他参数，包括：
                - lyrics: 歌词文本
                - audio_duration: 音频时长
                - infer_step: 推理步数
                - guidance_scale: 引导尺度
                - scheduler_type: 调度器类型
                - 等等
        """
        try:
            # 从kwargs中获取参数，如果没有则使用默认值
            params = self.default_params.copy()
            params.update(kwargs)

            # 处理必需参数
            prompt = input
            lyrics = kwargs.get("lyrics", "")

            # 生成输出路径
            output_filename = f"acestep_output_{uuid.uuid4().hex}.wav"

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, output_filename)

                # 调用AceStep模型进行推理，按照infer.py的参数顺序
                self._model(
                    audio_duration=params["audio_duration"],
                    prompt=prompt,
                    lyrics=lyrics,
                    infer_step=params["infer_step"],
                    guidance_scale=params["guidance_scale"],
                    scheduler_type=params["scheduler_type"],
                    cfg_type=params["cfg_type"],
                    omega_scale=params["omega_scale"],
                    manual_seeds=", ".join(map(str, params["actual_seeds"])),
                    guidance_interval=params["guidance_interval"],
                    guidance_interval_decay=params["guidance_interval_decay"],
                    min_guidance_scale=params["min_guidance_scale"],
                    use_erg_tag=params["use_erg_tag"],
                    use_erg_lyric=params["use_erg_lyric"],
                    use_erg_diffusion=params["use_erg_diffusion"],
                    oss_steps=", ".join(map(str, params["oss_steps"])),
                    guidance_scale_text=params["guidance_scale_text"],
                    guidance_scale_lyric=params["guidance_scale_lyric"],
                    save_path=output_path,
                )

                # 转换为指定格式
                final_output_path = convert(output_path, response_format, speed)
                return final_output_path

        except Exception as e:
            logger.error(f"Failed to generate speech with AceStep: {e}")
            raise

    def _get_voices(self) -> List[str]:
        """AceStep不使用预定义的声音，返回空列表"""
        return []
