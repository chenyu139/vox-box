import asyncio
import functools
import tempfile
import os
import torchaudio
from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse

from vox_box.backends.stt.base import STTBackend
from vox_box.backends.tts.base import TTSBackend
from vox_box.server.model import get_model_instance
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

executor = ThreadPoolExecutor(max_workers=1)


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert (
            sample_rate > target_sr
        ), "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


ALLOWED_SPEECH_OUTPUT_AUDIO_TYPES = {
    "mp3",
    "opus",
    "aac",
    "flac",
    "wav",
    "pcm",
}


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: str = "mp3"
    speed: float = 1.0


@router.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    try:
        if (
            request.response_format
            and request.response_format not in ALLOWED_SPEECH_OUTPUT_AUDIO_TYPES
        ):
            return HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {request.response_format}",
            )

        if request.speed < 0.25 or request.speed > 2:
            return HTTPException(
                status_code=400, detail="Speed must be between 0.25 and 2"
            )

        model_instance: TTSBackend = get_model_instance()
        if not isinstance(model_instance, TTSBackend):
            return HTTPException(
                status_code=400, detail="Model instance does not support speech API"
            )

        func = functools.partial(
            model_instance.speech,
            request.input,
            request.voice,
            request.speed,
            request.response_format,
        )

        loop = asyncio.get_event_loop()
        audio_file = await loop.run_in_executor(
            executor,
            func,
        )

        media_type = get_media_type(request.response_format)
        return FileResponse(audio_file, media_type=media_type)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to generate speech, {e}")


class SpeechInstructRequest(BaseModel):
    model: str
    input: str
    instruct_text: str
    response_format: str = "mp3"
    speed: float = 1.0


class AceStepRequest(BaseModel):
    model: str
    prompt: str
    lyrics: str = ""
    audio_duration: float = 10.0
    infer_step: int = 60  # 与ACE-Step原始实现一致
    guidance_scale: float = 15.0  # 与ACE-Step原始实现一致
    scheduler_type: str = "euler"
    cfg_type: str = "apg"  # 与ACE-Step原始实现一致
    omega_scale: float = 10.0  # 与ACE-Step原始实现一致
    guidance_interval: float = 0.5  # 与ACE-Step原始实现一致
    guidance_interval_decay: float = 0.0
    min_guidance_scale: float = 3.0  # 与ACE-Step原始实现一致
    use_erg_tag: bool = True
    use_erg_lyric: bool = True
    use_erg_diffusion: bool = True
    oss_steps: list = []  # 与ACE-Step原始实现一致
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0
    actual_seeds: list = [42]
    response_format: str = "mp3"


@router.post("/v1/audio/speech_instruct")
async def speech_instruct(request: Request):
    try:
        form = await request.form()
        keys = form.keys()

        # 检查必需的字段
        required_fields = ["model", "input", "instruct_text", "voice"]
        for field in required_fields:
            if field not in keys:
                return HTTPException(
                    status_code=400, detail=f"Field {field} is required"
                )

        # 获取表单数据
        input_text = form.get("input")
        instruct_text = form.get("instruct_text")
        response_format = form.get("response_format", "mp3")
        speed = float(form.get("speed", 1.0))

        # 获取音频文件
        voice_file: UploadFile = form["voice"]
        if not voice_file:
            return HTTPException(status_code=400, detail="Voice audio file is required")

        # 检查音频格式
        file_content_type = voice_file.content_type
        if file_content_type not in ALLOWED_TRANSCRIPTIONS_INPUT_AUDIO_FORMATS:
            return HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file_content_type}",
            )

        # 检查输出格式
        if response_format not in ALLOWED_SPEECH_OUTPUT_AUDIO_TYPES:
            return HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {response_format}",
            )

        # 检查速度参数
        if speed < 0.25 or speed > 2:
            return HTTPException(
                status_code=400, detail="Speed must be between 0.25 and 2"
            )

        # 保存音频文件到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            voice_audio_bytes = await voice_file.read()
            temp_file.write(voice_audio_bytes)
            temp_audio_path = temp_file.name

        try:
            # 使用load_wav处理音频文件
            speech = load_wav(temp_audio_path, 16000)  # 假设目标采样率为16kHz

            model_instance: TTSBackend = get_model_instance()
            if not isinstance(model_instance, TTSBackend):
                return HTTPException(
                    status_code=400, detail="Model instance does not support speech API"
                )

            # 检查模型是否支持speech_instruct方法
            if not hasattr(model_instance, "speech_instruct"):
                return HTTPException(
                    status_code=400, detail="Model does not support speech_instruct API"
                )

            func = functools.partial(
                model_instance.speech_instruct,
                input_text,
                instruct_text,
                speech,  # 传递处理后的speech tensor而不是原始字节
                speed,
                response_format,
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

        loop = asyncio.get_event_loop()
        audio_file = await loop.run_in_executor(
            executor,
            func,
        )

        media_type = get_media_type(response_format)
        return FileResponse(audio_file, media_type=media_type)
    except Exception as e:
        return HTTPException(
            status_code=500, detail=f"Failed to generate speech_instruct, {e}"
        )


# ref: https://github.com/LMS-Community/slimserver/blob/public/10.0/types.conf
ALLOWED_TRANSCRIPTIONS_INPUT_AUDIO_FORMATS = {
    # flac
    "audio/flac",
    "audio/x-flac",
    # mp3
    "audio/mpeg",
    "audio/x-mpeg",
    "audio/mp3",
    "audio/mp3s",
    "audio/mpeg3",
    "audio/mpg",
    # mp4
    "audio/m4a",
    "audio/x-m4a",
    "audio/mp4",
    # mpeg
    "audio/mpga",
    # ogg
    "audio/ogg",
    "audio/x-ogg",
    # wav
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    # webm
    "video/webm",
    "audio/webm",
    # file
    "application/octet-stream",
}

ALLOWED_TRANSCRIPTIONS_OUTPUT_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}


@router.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    try:
        form = await request.form()
        keys = form.keys()
        if "file" not in keys:
            return HTTPException(status_code=400, detail="Field file is required")

        file: UploadFile = form[
            "file"
        ]  # flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm
        file_content_type = file.content_type
        if file_content_type not in ALLOWED_TRANSCRIPTIONS_INPUT_AUDIO_FORMATS:
            return HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_content_type}",
            )

        audio_bytes = await file.read()
        language = form.get("language")
        prompt = form.get("prompt")
        temperature = float(form.get("temperature", 0))
        if not (0 <= temperature <= 1):
            return HTTPException(
                status_code=400, detail="Temperature must be between 0 and 1"
            )

        timestamp_granularities = form.getlist("timestamp_granularities")
        response_format = form.get("response_format", "json")
        if response_format not in ALLOWED_TRANSCRIPTIONS_OUTPUT_FORMATS:
            return HTTPException(
                status_code=400, detail="Unsupported response_format: {response_format}"
            )

        model_instance: STTBackend = get_model_instance()
        if not isinstance(model_instance, STTBackend):
            return HTTPException(
                status_code=400,
                detail="Model instance does not support transcriptions API",
            )

        kwargs = {
            "content_type": file_content_type,
        }
        func = functools.partial(
            model_instance.transcribe,
            audio_bytes,
            language,
            prompt,
            temperature,
            timestamp_granularities,
            response_format,
            **kwargs,
        )

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            executor,
            func,
        )

        if response_format == "json":
            return {"text": data}
        elif response_format == "text":
            return data
        else:
            return data
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to transcribe audio, {e}")


@router.get("/health")
async def health():
    model_instance = get_model_instance()
    if model_instance is None or (not model_instance.is_load()):
        return HTTPException(status_code=503, detail="Loading model")
    return {"status": "ok"}


@router.get("/v1/models")
async def get_model_list():
    model_instance = get_model_instance()
    if model_instance is None:
        return []
    return {"object": "list", "data": [model_instance.model_info()]}


@router.get("/v1/models/{model_id}")
async def get_model_info(model_id: str):
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return model_instance.model_info()


@router.get("/v1/languages")
async def get_languages():
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return {
        "languages": model_instance.model_info().get("languages", []),
    }


@router.get("/v1/voices")
async def get_voice():
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return {
        "voices": model_instance.model_info().get("voices", []),
    }


@router.post("/v1/audio/acestep")
async def acestep_generate(request: AceStepRequest):
    """AceStep专用端点，用于生成背景音乐/音频"""
    try:
        if request.response_format not in ALLOWED_SPEECH_OUTPUT_AUDIO_TYPES:
            return HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {request.response_format}",
            )

        model_instance: TTSBackend = get_model_instance()
        if not isinstance(model_instance, TTSBackend):
            return HTTPException(
                status_code=400, detail="Model instance does not support TTS API"
            )

        # 检查是否为AceStep模型
        from vox_box.backends.tts.acestep import AceStep

        if not isinstance(model_instance, AceStep):
            return HTTPException(
                status_code=400, detail="This endpoint requires AceStep model"
            )

        # 准备参数
        kwargs = {
            "lyrics": request.lyrics,
            "audio_duration": request.audio_duration,
            "infer_step": request.infer_step,
            "guidance_scale": request.guidance_scale,
            "scheduler_type": request.scheduler_type,
            "cfg_type": request.cfg_type,
            "omega_scale": request.omega_scale,
            "guidance_interval": request.guidance_interval,
            "guidance_interval_decay": request.guidance_interval_decay,
            "min_guidance_scale": request.min_guidance_scale,
            "use_erg_tag": request.use_erg_tag,
            "use_erg_lyric": request.use_erg_lyric,
            "use_erg_diffusion": request.use_erg_diffusion,
            "oss_steps": request.oss_steps,
            "guidance_scale_text": request.guidance_scale_text,
            "guidance_scale_lyric": request.guidance_scale_lyric,
            "actual_seeds": request.actual_seeds,
        }

        func = functools.partial(
            model_instance.speech,
            request.prompt,
            None,  # voice参数对AceStep无意义
            1.0,  # speed参数对AceStep无意义
            request.response_format,
            **kwargs,
        )

        loop = asyncio.get_event_loop()
        audio_file = await loop.run_in_executor(
            executor,
            func,
        )

        media_type = get_media_type(request.response_format)
        return FileResponse(audio_file, media_type=media_type)
    except Exception as e:
        return HTTPException(
            status_code=500, detail=f"Failed to generate audio with AceStep: {e}"
        )


def get_media_type(response_format) -> str:
    if response_format == "mp3":
        media_type = "audio/mpeg"
    elif response_format == "opus":
        media_type = "audio/ogg;codec=opus"
    elif response_format == "aac":
        media_type = "audio/aac"
    elif response_format == "flac":
        media_type = "audio/x-flac"
    elif response_format == "wav":
        media_type = "audio/wav"
    elif response_format == "pcm":
        media_type = "audio/pcm"
    else:
        raise Exception(
            f"Invalid response_format: '{response_format}'", param="response_format"
        )

    return media_type
