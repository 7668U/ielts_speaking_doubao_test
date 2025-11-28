import asyncio
import json
import gzip
import logging
import uuid
import struct
import websockets

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASR_Service")

class VolcASRClient:
    def __init__(self, appid: str, token: str, cluster: str = "volc.bigasr.sauc.duration"):
        self.appid = appid
        self.token = token
        self.cluster = cluster
        # V3 流式接口地址
        self.ws_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
        self.ws = None
        self.session_id = str(uuid.uuid4())

    def _construct_header(self, msg_type, msg_flags, serial_method, compression_type):
        """构建 4 字节的固定 Header"""
        version = 0b0001
        header_size = 0b0001
        byte_0 = (version << 4) | header_size
        byte_1 = (msg_type << 4) | msg_flags
        byte_2 = (serial_method << 4) | compression_type
        byte_3 = 0x00
        return bytes([byte_0, byte_1, byte_2, byte_3])

    def _construct_payload(self, data, is_json=False):
        """构建 Payload"""
        if is_json:
            payload_bytes = json.dumps(data).encode('utf-8')
        else:
            payload_bytes = data 

        # Gzip 压缩
        compressed_payload = gzip.compress(payload_bytes)
        payload_size = len(compressed_payload)
        size_bytes = struct.pack('!I', payload_size)
        return size_bytes, compressed_payload

    async def connect(self):
        """建立 WebSocket 连接"""
        headers = {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.token,
            "X-Api-Resource-Id": self.cluster,
            "X-Api-Connect-Id": self.session_id
        }
        try:
            # 兼容不同版本的 websockets
            try:
                self.ws = await websockets.connect(self.ws_url, additional_headers=headers)
            except TypeError:
                self.ws = await websockets.connect(self.ws_url, extra_headers=headers)
            
            logger.info(f"ASR Connected: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"ASR Connection failed: {e}")
            return False

    async def send_full_client_request(self):
        """
        发送初始配置包 (V3 协议格式)
        注意：V3 协议不支持 language 参数，它会自动检测。
        只要你的发音是英文，且前端降采样正常，它就会识别为英文。
        """
        req_params = {
            "user": {"uid": "ielts_candidate"},
            "audio": {
                "format": "pcm",  
                "rate": 16000,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,   
                "enable_punc": True,  
            }
        }
        # Header: Full Client Request (0b0001)
        header = self._construct_header(0b0001, 0b0000, 0b0001, 0b0001)
        size_bytes, payload = self._construct_payload(req_params, is_json=True)
        await self.ws.send(header + size_bytes + payload)
        logger.info("Sent Full Client Request (V3)")

    async def send_audio_chunk(self, audio_data: bytes, is_last: bool = False):
        """发送音频数据包"""
        flags = 0b0010 if is_last else 0b0000
        header = self._construct_header(0b0010, flags, 0b0000, 0b0001)
        size_bytes, payload = self._construct_payload(audio_data, is_json=False)
        await self.ws.send(header + size_bytes + payload)

    async def receive_result(self):
        """解析服务端返回的数据 (带去重)"""
        last_text = ""
        while True:
            try:
                resp = await self.ws.recv()
                
                header = resp[:4]
                msg_type = (header[1] >> 4) & 0b1111
                compression_type = header[2] & 0b00001111

                if msg_type == 0b1111: # Error
                    logger.error("Received Error Frame")
                    break 

                payload_size = struct.unpack('!I', resp[8:12])[0]
                payload_raw = resp[12 : 12 + payload_size]
                
                try:
                    if compression_type == 0b0001: # Gzip
                        payload_json = gzip.decompress(payload_raw)
                    else: # 无压缩
                        payload_json = payload_raw
                    
                    result = json.loads(payload_json)
                    
                    # 提取文本并去重
                    current_text = ""
                    if 'result' in result:
                        if 'text' in result['result']:
                             current_text = result['result']['text']
                        elif isinstance(result['result'], list):
                             for item in result['result']:
                                 if 'text' in item:
                                     current_text = item['text']
                    
                    # 输出逻辑
                    if current_text and current_text.strip() and current_text != last_text:
                        logger.info(f"ASR Result: {current_text}")
                        yield current_text
                        last_text = current_text
                    
                except Exception as e:
                    logger.error(f"Payload parse error: {e}")

            except websockets.exceptions.ConnectionClosed:
                logger.info("ASR Connection Closed")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break