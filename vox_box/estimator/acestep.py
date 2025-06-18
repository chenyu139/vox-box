import os
import logging
from typing import Dict

from vox_box.estimator.base import Estimator
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.utils.model import create_model_dict

logger = logging.getLogger(__name__)


class AceStep(Estimator):
    def __init__(self, cfg: Config):
        self._cfg = cfg
        # AceStep模型的必需文件，基于third_party/ACE-Step的结构
        self._required_files = [
            "acestep",  # acestep模块目录
            "pipeline_ace_step.py",  # 主要的pipeline文件
        ]
        # 或者检查特定的配置文件
        self._config_files = [
            "config.yaml",
            "model_config.json",
            "checkpoint",
        ]

    def model_info(self) -> Dict:
        model = (
            self._cfg.model
            or self._cfg.huggingface_repo_id
            or self._cfg.model_scope_model_id
        )
        supported = self._supported()
        return create_model_dict(
            model,
            supported=supported,
            task_type=TaskTypeEnum.TTS,
            backend_framework=BackendEnum.ACESTEP,
        )

    def _supported(self) -> bool:
        if self._cfg.model is not None:
            return self._check_local_model(self._cfg.model)
        elif (
            self._cfg.huggingface_repo_id is not None
            or self._cfg.model_scope_model_id is not None
        ):
            return self._check_remote_model()
        return False

    def _check_local_model(self, base_dir: str) -> bool:
        # 检查是否存在AceStep相关文件
        for required_file in self._required_files:
            file_path = os.path.join(base_dir, required_file)
            if os.path.exists(file_path):
                return True

        # 检查配置文件
        for config_file in self._config_files:
            file_path = os.path.join(base_dir, config_file)
            if os.path.exists(file_path):
                return True

        # 检查是否包含AceStep关键词的文件
        try:
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if "acestep" in file.lower() or "ace_step" in file.lower():
                        return True
                for dir_name in dirs:
                    if "acestep" in dir_name.lower() or "ace_step" in dir_name.lower():
                        return True
        except Exception as e:
            logger.warning(f"Error checking AceStep model directory: {e}")

        return False

    def _check_remote_model(self) -> bool:
        # 对于远程模型，可以检查repo_id或model_id是否包含acestep关键词
        repo_id = self._cfg.huggingface_repo_id or self._cfg.model_scope_model_id
        if repo_id and ("acestep" in repo_id.lower() or "ace-step" in repo_id.lower()):
            return True
        return False
