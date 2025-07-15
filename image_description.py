"""
图像描述模块 - 基于蓝心模型的医学图像描述功能
"""
import base64
import uuid
import time
import requests
import os
from auth_util import gen_sign_headers


class ImageDescriptionService:
    """图像描述服务类"""
    
    def __init__(self, app_id='2025630384', app_key='fsUlhWWiDgeCqEfi'):
        """
        初始化图像描述服务
        
        Args:
            app_id: 蓝心模型APP_ID
            app_key: 蓝心模型APP_KEY
        """
        self.app_id = app_id
        self.app_key = app_key
        self.uri = '/vivogpt/completions'
        self.domain = 'api-ai.vivo.com.cn'
        self.method = 'POST'
        self.model = 'vivo-BlueLM-V-2.0'
    
    def encode_image_to_base64(self, image_path):
        """
        将图像文件编码为base64格式
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: base64编码的图像数据
        """
        try:
            with open(image_path, "rb") as f:
                b_image = f.read()
            return base64.b64encode(b_image).decode('utf-8')
        except Exception as e:
            print(f"图像编码失败: {e}")
            return None
    
    def describe_medical_image(self, segmented_image_path):
        """
        描述医学图像（仅处理分割后的图像）
        
        Args:
            segmented_image_path: 分割后图像路径
            
        Returns:
            tuple: (成功标志, 描述文本或错误信息)
        """
        try:
            # 检查分割后图像是否存在
            if not segmented_image_path or not os.path.exists(segmented_image_path):
                return False, "分割后图像不存在"
            
            # 编码分割后图像
            segmented_image_b64 = self.encode_image_to_base64(segmented_image_path)
            if not segmented_image_b64:
                return False, "分割后图像编码失败"
            
            # 构建消息内容
            messages = []
            
            # 添加分割后图像
            messages.append({
                "role": "user",
                "content": "data:image/JPEG;base64," + segmented_image_b64,
                "contentType": "image"
            })
            
            # 添加医学描述提示
            prompt_text = """请作为一名专业的医学影像专家，分析这张经过图像分割处理的医学影像。

请从以下几个方面进行专业分析：
1. 影像类型识别：判断这是什么类型的医学影像（如X光、CT、MRI、超声、病理切片等）
2. 分割结果分析：识别图像分割后突出显示的主要解剖结构和区域
3. 医学结构识别：识别图像中的重要医学结构和器官
4. 异常发现：如果存在异常区域，请指出可能的病变或异常表现
5. 临床意义：基于分割结果和影像表现，说明潜在的临床意义

请用专业但易懂的语言进行描述，为医学诊断提供有价值的参考信息。"""
            
            messages.append({
                "role": "user",
                "content": prompt_text,
                "contentType": "text"
            })
            
            # 构建请求参数
            params = {
                'requestId': str(uuid.uuid4())
            }
            
            data = {
                'prompt': '医学影像分析',
                'sessionId': str(uuid.uuid4()),
                'requestId': params['requestId'],
                'model': self.model,
                "messages": messages,
            }
            
            # 生成请求头
            headers = gen_sign_headers(self.app_id, self.app_key, self.method, self.uri, params)
            headers['Content-Type'] = 'application/json'
            
            # 发送请求
            start_time = time.time()
            url = f'http://{self.domain}{self.uri}'
            response = requests.post(url, json=data, headers=headers, params=params, timeout=30)
            
            end_time = time.time()
            timecost = end_time - start_time
            print(f"蓝心模型请求耗时: {timecost:.2f}秒")
            
            if response.status_code == 200:
                result = response.json()
                print(f"蓝心模型响应: {result}")
                
                # 提取描述文本
                if 'data' in result and 'content' in result['data']:
                    description = result['data']['content']
                    return True, description
                elif 'choices' in result and len(result['choices']) > 0:
                    description = result['choices'][0].get('message', {}).get('content', '')
                    return True, description
                else:
                    return False, "响应格式异常，无法提取描述内容"
            else:
                error_msg = f"请求失败: {response.status_code} {response.text}"
                print(error_msg)
                return False, error_msg
                
        except requests.exceptions.Timeout:
            return False, "请求超时，请稍后重试"
        except requests.exceptions.RequestException as e:
            return False, f"网络请求异常: {str(e)}"
        except Exception as e:
            return False, f"图像描述生成失败: {str(e)}"
    
    def describe_single_image(self, image_path, custom_prompt=None):
        """
        描述单张图像
        
        Args:
            image_path: 图像文件路径
            custom_prompt: 自定义提示词
            
        Returns:
            tuple: (成功标志, 描述文本或错误信息)
        """
        try:
            # 编码图像
            image_b64 = self.encode_image_to_base64(image_path)
            if not image_b64:
                return False, "图像编码失败"
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": "data:image/JPEG;base64," + image_b64,
                    "contentType": "image"
                },
                {
                    "role": "user",
                    "content": custom_prompt or "请详细描述这张医学影像的内容和特征。",
                    "contentType": "text"
                }
            ]
            
            # 构建请求参数
            params = {
                'requestId': str(uuid.uuid4())
            }
            
            data = {
                'prompt': '医学影像描述',
                'sessionId': str(uuid.uuid4()),
                'requestId': params['requestId'],
                'model': self.model,
                "messages": messages,
            }
            
            # 生成请求头
            headers = gen_sign_headers(self.app_id, self.app_key, self.method, self.uri, params)
            headers['Content-Type'] = 'application/json'
            
            # 发送请求
            url = f'http://{self.domain}{self.uri}'
            response = requests.post(url, json=data, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # 提取描述文本
                if 'data' in result and 'content' in result['data']:
                    description = result['data']['content']
                    return True, description
                elif 'choices' in result and len(result['choices']) > 0:
                    description = result['choices'][0].get('message', {}).get('content', '')
                    return True, description
                else:
                    return False, "响应格式异常，无法提取描述内容"
            else:
                return False, f"请求失败: {response.status_code} {response.text}"
                
        except Exception as e:
            return False, f"图像描述生成失败: {str(e)}"


# 全局图像描述服务实例
image_description_service = ImageDescriptionService()


if __name__ == "__main__":
    # 测试代码
    service = ImageDescriptionService()
    print("图像描述服务初始化完成") 