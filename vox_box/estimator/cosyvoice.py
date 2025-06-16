import logging
import os
from typing import Dict

from vox_box.downloader.downloaders import download_file
from vox_box.estimator.base import Estimator

from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.utils.model import create_model_dict

logger = logging.getLogger(__name__)


class CosyVoice(Estimator):
    def __init__(
        self,
        cfg: Config,
    ):
        self._cfg = cfg
        self._required_files = [
            "cosyvoice.yaml",
            "cosyvoice2.yaml",  # 添加对新文件名的支持
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
            backend_framework=BackendEnum.COSY_VOICE,
        )

    def _supported(self) -> bool:
        if self._cfg.model is not None:
            return self._check_local_model(self._cfg.model)
        elif (
            self._cfg.huggingface_repo_id is not None
            or self._cfg.model_scope_model_id is not None
        ):
            return self._check_remote_model()

    def _check_local_model(self, base_dir: str) -> bool:
        # 检查是否存在任一配置文件
        for required_file in self._required_files:
            if os.path.exists(os.path.join(base_dir, required_file)):
                return True
        return False

    def _check_remote_model(self) -> bool:
        downloaded_files = []
        for f in self._required_files:
            try:
                download_file_path = download_file(
                    huggingface_repo_id=self._cfg.huggingface_repo_id,
                    huggingface_filename=f,
                    model_scope_model_id=self._cfg.model_scope_model_id,
                    model_scope_file_path=f,
                    cache_dir=self._cfg.cache_dir,
                )
            except Exception as e:
                logger.debug(f"File {f} does not exist, {e}")
                continue
            downloaded_files.append(download_file_path)

        if len(downloaded_files) != 0:
            base_dir = os.path.dirname(downloaded_files[0])
            return self._check_local_model(base_dir)

        return False
