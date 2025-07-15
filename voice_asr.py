#!/usr/bin/env python
# -*- coding: utf-8 -*-

import soundfile
import struct
import json
import time
import tempfile
import os
import numpy as np
from websocket import create_connection
from urllib import parse
from auth_util import gen_sign_headers

class VoiceASR:
    """蓝心大模型语音识别工具"""
    
    def __init__(self):
        self.appid = '2025630384'
        self.appkey = 'fsUlhWWiDgeCqEfi'
        self.domain = 'api-ai.vivo.com.cn'
        
    def recognize_audio_file(self, audio_file_path: str) -> str:
        """
        识别音频文件并返回文本结果
        
        Args:
            audio_file_path: 音频文件路径
            
        Returns:
            识别出的文本内容
        """
        try:
            # 读取音频数据并转换格式
            wav_data = self._load_and_convert_audio(audio_file_path)
            if wav_data is None:
                return ""
            
            # 建立WebSocket连接
            ws = self._create_websocket_connection()
            
            # 发送音频数据并获取结果
            result_text = self._process_audio_data(ws, wav_data)
            
            ws.close()
            return result_text
            
        except Exception as e:
            print(f"语音识别错误: {str(e)}")
            return ""
    
    def _load_and_convert_audio(self, audio_file_path: str):
        """
        加载音频文件并转换为蓝心ASR要求的格式
        要求：16k采样率，16位，单声道，PCM编码
        """
        try:
            # 读取音频文件
            wav_data, sample_rate = soundfile.read(audio_file_path)
            
            print(f"原始音频格式: 采样率={sample_rate}Hz, 数据类型={wav_data.dtype}, 形状={wav_data.shape}")
            
            # 如果是多声道，转换为单声道
            if len(wav_data.shape) > 1:
                print("检测到多声道音频，转换为单声道...")
                wav_data = np.mean(wav_data, axis=1)
            
            # 转换采样率为16kHz
            if sample_rate != 16000:
                print(f"采样率从 {sample_rate}Hz 转换为 16000Hz...")
                # 简单的重采样：使用线性插值
                duration = len(wav_data) / sample_rate
                new_length = int(duration * 16000)
                wav_data = np.interp(
                    np.linspace(0, len(wav_data) - 1, new_length),
                    np.arange(len(wav_data)),
                    wav_data
                )
            
            # 转换为int16格式
            if wav_data.dtype != np.int16:
                print(f"数据类型从 {wav_data.dtype} 转换为 int16...")
                # 如果是浮点数，先归一化到[-1, 1]，然后转换为int16
                if wav_data.dtype in [np.float32, np.float64]:
                    # 确保数据在[-1, 1]范围内
                    wav_data = np.clip(wav_data, -1.0, 1.0)
                    # 转换为int16
                    wav_data = (wav_data * 32767).astype(np.int16)
                else:
                    # 其他整数类型直接转换
                    wav_data = wav_data.astype(np.int16)
            
            print(f"转换后格式: 采样率=16000Hz, 数据类型=int16, 长度={len(wav_data)}")
            return wav_data
            
        except Exception as e:
            print(f"音频格式转换错误: {str(e)}")
            print("请确保上传的是有效的音频文件")
            return None
    
    def _create_websocket_connection(self):
        """创建WebSocket连接"""
        t = int(round(time.time() * 1000))
        
        params = {
            'client_version': parse.quote('unknown'),
            'product': parse.quote('x'),
            'package': parse.quote('unknown'),
            'sdk_version': parse.quote('unknown'),
            'user_id': parse.quote('2addc42b7ae689dfdf1c63e220df52a2'),
            'android_version': parse.quote('unknown'),
            'system_time': parse.quote(str(t)),
            'net_type': 1,
            'engineid': "shortasrinput"
        }
        
        headers = gen_sign_headers(self.appid, self.appkey, 'GET', '/asr/v2', params)
        
        param_str = '&'.join([f"{key}={value}" for key, value in params.items()])
        ws_url = f'ws://{self.domain}/asr/v2?{param_str}'
        
        return create_connection(ws_url, header=headers)
    
    def _process_audio_data(self, ws, wav_data) -> str:
        """处理音频数据并获取识别结果"""
        result_text = ""
        
        try:
            # 发送开始信号
            start_data = {
                "type": "started",
                "request_id": "req_id",
                "asr_info": {
                    "front_vad_time": 6000,
                    "end_vad_time": 2000,
                    "audio_type": "pcm",
                    "chinese2digital": 1,
                    "punctuation": 2,
                },
                "business_info": "{\"scenes_pkg\":\"com.tencent.qqlive\", \"editor_type\":\"3\", \"pro_id\":\"2addc42b7ae689dfdf1c63e220df52a2-2020\"}"
            }
            
            ws.send(json.dumps(start_data))
            
            # 准备音频数据
            nlen = len(wav_data)
            nframes = nlen * 2
            pack_data = struct.pack('%dh' % nlen, *wav_data)
            wav_data_c = list(struct.unpack('B' * nframes, pack_data))
            
            # 分块发送音频数据
            cur_frames = 0
            sample_frames = 1280
            
            while cur_frames < nframes:
                samp_remaining = nframes - cur_frames
                num_samp = sample_frames if sample_frames < samp_remaining else samp_remaining
                
                list_tmp = [None] * num_samp
                for j in range(num_samp):
                    list_tmp[j] = wav_data_c[cur_frames + j]
                
                pack_data_2 = struct.pack('%dB' % num_samp, *list_tmp)
                cur_frames += num_samp
                
                if len(pack_data_2) < 1280:
                    break
                
                ws.send_binary(pack_data_2)
                time.sleep(0.04)
            
            # 发送结束信号
            ws.send_binary(b'--end--')
            
            # 接收并处理结果
            result_text = self._receive_results(ws)
            
            # 发送关闭信号
            ws.send_binary(b'--close--')
            
        except Exception as e:
            print(f"处理音频数据时出错: {str(e)}")
        
        return result_text
    
    def _receive_results(self, ws) -> str:
        """接收识别结果"""
        result_text = ""
        
        while True:
            try:
                response = ws.recv()
                data = json.loads(response)
                
                if data["action"] == "error":
                    print(f"ASR错误: {data}")
                    break
                    
                if data["action"] == "result" and data["type"] == "asr":
                    if data["data"]["is_last"] is True:
                        result_text = data["data"]["text"]
                        break
                        
            except Exception as e:
                print(f"接收结果时出错: {str(e)}")
                break
        
        return result_text

def recognize_voice(audio_file_path: str) -> str:
    """
    简单的语音识别接口函数
    
    Args:
        audio_file_path: 音频文件路径
        
    Returns:
        识别出的文本
    """
    asr = VoiceASR()
    return asr.recognize_audio_file(audio_file_path)

if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        result = recognize_voice(audio_path)
        print(f"识别结果: {result}")
    else:
        print("用法: python voice_asr.py <音频文件路径>") 