#!/usr/bin/env python3
"""
验证WAV文件中的AIGC元数据
"""

import os
import sys
import struct
import json


def extract_aigc_metadata_from_wav(wav_file_path: str):
    """从WAV文件中提取AIGC元数据"""
    try:
        with open(wav_file_path, "rb") as f:
            data = f.read()

        # 检查是否为有效的WAV文件
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise ValueError("不是有效的WAV文件")

        # 查找INFO chunk中的AIGC数据
        pos = 12
        while pos < len(data) - 8:
            chunk_id = data[pos : pos + 4]
            chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]

            if chunk_id == b"LIST":
                # 检查是否为INFO LIST
                list_type = data[pos + 8 : pos + 12]
                if list_type == b"INFO":
                    # 在INFO chunk中查找AIGC
                    info_pos = pos + 12
                    info_end = pos + 8 + chunk_size

                    while info_pos < info_end - 8:
                        info_chunk_id = data[info_pos : info_pos + 4]
                        info_chunk_size = struct.unpack(
                            "<I", data[info_pos + 4 : info_pos + 8]
                        )[0]

                        if info_chunk_id == b"AIGC":
                            # 找到AIGC数据
                            aigc_data = data[
                                info_pos + 8 : info_pos + 8 + info_chunk_size
                            ]
                            # 移除可能的填充字节
                            aigc_data = aigc_data.rstrip(b"\x00")
                            return aigc_data.decode("utf-8")

                        # 移动到下一个info chunk
                        info_pos += 8 + info_chunk_size
                        if info_chunk_size % 2 == 1:
                            info_pos += 1

            # 移动到下一个chunk
            pos += 8 + chunk_size
            if chunk_size % 2 == 1:
                pos += 1

        return None

    except Exception as e:
        print(f"提取AIGC元数据时出错: {e}")
        return None


def verify_aigc_compliance(aigc_json_str: str):
    """验证AIGC元数据是否符合合规要求"""
    try:
        # 解析JSON
        aigc_data = json.loads(aigc_json_str)

        # 检查必需的结构
        if "AIGC" not in aigc_data:
            return False, "缺少AIGC字段"

        aigc_content = aigc_data["AIGC"]
        required_fields = [
            "Label",
            "ContentProducer",
            "ProduceID",
            "ReservedCode1",
            "ContentPropagator",
            "PropagateID",
            "ReservedCode2",
        ]

        # 检查所有必需字段
        for field in required_fields:
            if field not in aigc_content:
                return False, f"缺少必需字段: {field}"

        # 验证Label值
        label = aigc_content["Label"]
        if label not in ["1", "2", "3"]:
            return False, f"Label值无效: {label}，应为1、2或3"

        # 验证首次生成的一致性要求
        if (
            aigc_content["ContentProducer"] == aigc_content["ContentPropagator"]
            and aigc_content["ProduceID"] == aigc_content["PropagateID"]
        ):
            consistency_check = "✅ 首次生成一致性检查通过"
        else:
            consistency_check = "⚠️  非首次生成或一致性检查未通过"

        return True, f"合规检查通过 - {consistency_check}"

    except json.JSONDecodeError as e:
        return False, f"JSON解析失败: {e}"
    except Exception as e:
        return False, f"验证过程出错: {e}"


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python verify_aigc_metadata.py <wav_file_path>")
        print("示例: python verify_aigc_metadata.py test_audio.wav")
        sys.exit(1)

    wav_file = sys.argv[1]

    if not os.path.exists(wav_file):
        print(f"错误: 文件不存在 - {wav_file}")
        sys.exit(1)

    print(f"=== 验证WAV文件AIGC元数据: {wav_file} ===")

    # 提取AIGC元数据
    aigc_metadata = extract_aigc_metadata_from_wav(wav_file)

    if aigc_metadata is None:
        print("❌ 未找到AIGC元数据")
        sys.exit(1)

    print(f"✅ 成功提取AIGC元数据:")
    print(f"原始数据: {aigc_metadata}")
    print()

    # 验证合规性
    is_compliant, message = verify_aigc_compliance(aigc_metadata)

    if is_compliant:
        print(f"✅ {message}")

        # 解析并显示详细信息
        try:
            aigc_data = json.loads(aigc_metadata)
            aigc_content = aigc_data["AIGC"]

            print("\n=== AIGC元数据详细信息 ===")
            label_meanings = {
                "1": "属于人工智能生成合成内容",
                "2": "可能为人工智能生成合成内容",
                "3": "疑似为人工智能生成合成内容",
            }

            print(
                f"生成合成标签 (Label): {aigc_content['Label']} - {label_meanings.get(aigc_content['Label'], '未知')}"
            )
            print(
                f"生成合成服务提供者 (ContentProducer): {aigc_content['ContentProducer']}"
            )
            print(f"内容制作编号 (ProduceID): {aigc_content['ProduceID']}")
            print(f"预留字段1 (ReservedCode1): '{aigc_content['ReservedCode1']}'")
            print(
                f"内容传播服务提供者 (ContentPropagator): {aigc_content['ContentPropagator']}"
            )
            print(f"内容传播编号 (PropagateID): {aigc_content['PropagateID']}")
            print(f"预留字段2 (ReservedCode2): '{aigc_content['ReservedCode2']}'")

        except Exception as e:
            print(f"⚠️  解析详细信息时出错: {e}")
    else:
        print(f"❌ {message}")
        sys.exit(1)

    print("\n=== 验证完成 ===")


if __name__ == "__main__":
    main()
