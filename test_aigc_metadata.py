#!/usr/bin/env python3
"""
测试AIGC元数据功能的简单脚本
"""

import os
import wave
import numpy as np
import json
import uuid
import struct


def add_aigc_metadata_to_wav(wav_file_path: str):
    """为WAV文件添加AIGC合规元数据"""
    try:
        # 尝试使用mutagen库
        from mutagen.wave import WAVE
        from mutagen.id3 import ID3NoHeaderError

        # 生成唯一ID
        produce_id = str(uuid.uuid4())
        propagate_id = produce_id  # 首次生成时，传播ID与制作ID一致

        # 构建AIGC元数据
        aigc_metadata = {
            "AIGC": {
                "Label": "1",  # 1=属于人工智能生成合成内容
                "ContentProducer": "VoxBox-TTS",
                "ProduceID": produce_id,
                "ReservedCode1": "",
                "ContentPropagator": "VoxBox-TTS",  # 首次生成时与ContentProducer一致
                "PropagateID": propagate_id,  # 首次生成时与ProduceID一致
                "ReservedCode2": "",
            }
        }

        # 转换为JSON字符串
        aigc_json = json.dumps(aigc_metadata, ensure_ascii=False, separators=(",", ":"))

        try:
            # 加载WAV文件
            audio_file = WAVE(wav_file_path)

            # 添加AIGC字段
            audio_file["AIGC"] = [aigc_json]

            # 保存文件
            audio_file.save()
            print(f"✅ 使用mutagen成功添加AIGC元数据")

        except Exception as e:
            print(f"⚠️  mutagen方法失败: {e}")
            # 如果mutagen失败，使用备用方法（直接修改WAV文件的INFO chunk）
            _add_aigc_info_chunk(wav_file_path, aigc_json)

    except ImportError:
        # 如果没有mutagen库，使用备用方法
        produce_id = str(uuid.uuid4())
        propagate_id = produce_id

        aigc_metadata = {
            "AIGC": {
                "Label": "1",
                "ContentProducer": "VoxBox-TTS",
                "ProduceID": produce_id,
                "ReservedCode1": "",
                "ContentPropagator": "VoxBox-TTS",
                "PropagateID": propagate_id,
                "ReservedCode2": "",
            }
        }

        aigc_json = json.dumps(aigc_metadata, ensure_ascii=False, separators=(",", ":"))
        _add_aigc_info_chunk(wav_file_path, aigc_json)


def _add_aigc_info_chunk(wav_file_path: str, aigc_json: str):
    """备用方法：直接修改WAV文件添加INFO chunk中的AIGC信息"""
    try:
        with open(wav_file_path, "rb") as f:
            data = bytearray(f.read())

        # 检查是否为有效的WAV文件
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise ValueError("不是有效的WAV文件")

        # 查找fmt chunk的结束位置
        pos = 12
        while pos < len(data) - 8:
            chunk_id = data[pos : pos + 4]
            chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]

            if chunk_id == b"data":
                # 在data chunk之前插入INFO chunk
                break

            # 移动到下一个chunk（确保偶数对齐）
            pos += 8 + chunk_size
            if chunk_size % 2 == 1:
                pos += 1

        # 创建AIGC INFO chunk
        aigc_data = aigc_json.encode("utf-8")

        # 构建INFO chunk
        info_chunk = b"LIST"
        info_size = 4 + 4 + 4 + len(aigc_data)  # INFO + AIGC + size + data
        if info_size % 2 == 1:
            info_size += 1
            aigc_data += b"\x00"  # 填充到偶数字节

        info_chunk += struct.pack("<I", info_size)
        info_chunk += b"INFO"
        info_chunk += b"AIGC"
        info_chunk += struct.pack("<I", len(aigc_data))
        info_chunk += aigc_data

        # 插入INFO chunk
        data[pos:pos] = info_chunk

        # 更新RIFF chunk的总大小
        new_riff_size = len(data) - 8
        data[4:8] = struct.pack("<I", new_riff_size)

        # 写回文件
        with open(wav_file_path, "wb") as f:
            f.write(data)

        print(f"✅ 使用备用方法成功添加AIGC元数据")

    except Exception as e:
        print(f"❌ 备用方法失败: {e}")
        raise


def create_test_wav_file(filename: str):
    """创建一个测试用的WAV文件"""
    # 生成1秒的440Hz正弦波（A音符）
    sample_rate = 22050
    duration = 1.0  # 秒
    frequency = 440.0  # Hz

    # 生成音频数据
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)

    # 转换为16位整数
    audio_data = (audio_data * 32767).astype(np.int16)

    # 保存为WAV文件
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16位
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"创建测试WAV文件: {os.path.abspath(filename)}")


def test_aigc_metadata():
    """测试AIGC元数据功能"""
    test_file = "test_audio.wav"

    try:
        # 创建测试WAV文件
        create_test_wav_file(test_file)

        # 获取原始文件大小
        original_size = os.path.getsize(test_file)
        print(f"原始文件大小: {original_size} 字节")

        # 添加AIGC元数据
        print("添加AIGC合规元数据...")
        add_aigc_metadata_to_wav(test_file)

        # 验证文件是否存在且大小合理
        if os.path.exists(test_file):
            new_size = os.path.getsize(test_file)
            print(
                f"处理后的文件大小: {new_size} 字节 (增加了 {new_size - original_size} 字节)"
            )

            # 尝试读取WAV文件以验证格式仍然有效
            with wave.open(test_file, "rb") as wf:
                frames = wf.getnframes()
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()

                print(f"WAV文件信息:")
                print(f"  - 帧数: {frames}")
                print(f"  - 采样率: {sample_rate} Hz")
                print(f"  - 声道数: {channels}")
                print(f"  - 采样位宽: {sample_width} 字节")
                print(f"  - 时长: {frames / sample_rate:.2f} 秒")

            print("✅ AIGC元数据添加成功！WAV文件格式有效。")

            # 尝试使用mutagen读取元数据（如果可用）
            try:
                from mutagen.wave import WAVE

                audio_file = WAVE(test_file)
                if "AIGC" in audio_file:
                    aigc_data = audio_file["AIGC"][0]
                    print(f"✅ 检测到AIGC元数据: {aigc_data}")

                    # 尝试解析JSON
                    try:
                        aigc_json = json.loads(aigc_data)
                        print(f"✅ AIGC元数据JSON解析成功:")
                        for key, value in aigc_json["AIGC"].items():
                            print(f"     {key}: {value}")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  AIGC元数据JSON解析失败: {e}")
                else:
                    print("ℹ️  使用mutagen未检测到AIGC字段，可能使用了备用方法")
            except ImportError:
                print("ℹ️  mutagen库不可用，使用了备用元数据方法")
            except Exception as e:
                print(f"⚠️  读取元数据时出错: {e}")

        else:
            print("❌ 处理后的文件不存在")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 保留测试文件用于检查
        if os.path.exists(test_file):
            print(f"保留测试文件用于检查: {os.path.abspath(test_file)}")


if __name__ == "__main__":
    print("=== AIGC元数据功能测试 ===")
    test_aigc_metadata()
    print("=== 测试完成 ===")
