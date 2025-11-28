import asyncio
import logging
import json
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from asr_service import VolcASRClient
from llm_service import DeepSeekClient

# ================= 配置区域 =================
ASR_APPID = "7407835878"
ASR_TOKEN = "Q3tgMiF7q_crzVMR0kPv8Jbk1JHF5kx7"
ASR_CLUSTER = "volc.bigasr.sauc.duration" 

# 请填入你的 DeepSeek Key
DEEPSEEK_API_KEY = "sk-e357919534ea4b098ec864c29d2166fc" 

IELTS_PROMPT = """
You are a friendly and professional IELTS Speaking Examiner. 
- Conduct a Part 1 interview.
- Start by asking for the candidate's full name.
- Ask only ONE simple question at a time.
- Topics: Hobbies, Hometown, Work/Study.
- Keep responses concise (1-2 sentences).
"""
# ===========================================

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IELTS_Server")

@app.get("/")
async def get():
    with open("index.html", "r", encoding='utf-8') as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(">>> 前端 WebSocket 已连接 (New Session)")

    llm_client = DeepSeekClient(api_key=DEEPSEEK_API_KEY, system_prompt=IELTS_PROMPT)
    asr_client = VolcASRClient(appid=ASR_APPID, token=ASR_TOKEN, cluster=ASR_CLUSTER)
    
    is_user_turn = True 
    current_full_text = ""      
    last_committed_length = 0   
    
    # --- ASR 连接逻辑 ---
    async def ensure_asr_connected():
        try:
            # 只要 ws 对象还在，我们就认为它是通的
            if asr_client.ws is not None:
                return True
            
            logger.info(">>> 正在建立 ASR 连接...")
            connected = await asr_client.connect()
            if connected:
                await asr_client.send_full_client_request()
                logger.info(">>> ASR 服务已就绪")
            return connected
        except Exception as e:
            logger.error(f"ASR 连接尝试失败: {e}")
            return False

    await ensure_asr_connected()

    # --- 任务 A: 接收前端消息 (音频 + 指令) ---
    async def receive_from_frontend():
        nonlocal is_user_turn, current_full_text, last_committed_length
        try:
            while True:
                message = await websocket.receive()

                # 1. 音频处理
                if "bytes" in message and message["bytes"]:
                    if is_user_turn:
                        try:
                            # 如果连接是 None，说明之前超时断开了，现在用户说话了，立刻重连！
                            if asr_client.ws is None: 
                                logger.info(">>> 检测到用户语音，正在从超时中恢复...")
                                raise Exception("Need Reconnect")
                            
                            await asr_client.send_audio_chunk(message["bytes"], is_last=False)
                        except Exception:
                            # 触发重连逻辑
                            asr_client.ws = None 
                            if await ensure_asr_connected(): 
                                current_full_text = ""
                                last_committed_length = 0
                                # 补发当前帧
                                try:
                                    await asr_client.send_audio_chunk(message["bytes"], is_last=False)
                                except: pass
                
                # 2. 文本指令处理
                elif "text" in message and message["text"]:
                    try:
                        data = json.loads(message["text"])
                        
                        if data.get("type") == "done_speaking":
                            if not is_user_turn: continue 

                            final_text = current_full_text[last_committed_length:].strip()
                            last_committed_length = len(current_full_text)

                            # 过滤空语音
                            if not final_text or final_text == "...": 
                                logger.info("未检测到有效输入 (Silence)，继续等待...")
                                await websocket.send_json({"type": "status_change", "status": "listening", "message": "Listening..."})
                                continue 

                            logger.info(f"【用户输入】: {final_text}")
                            
                            is_user_turn = False
                            
                            # Thinking
                            await websocket.send_json({"type": "status_change", "status": "processing", "message": f"You: {final_text}"})
                            
                            # LLM
                            ai_reply = await llm_client.get_response(final_text)
                            logger.info(f"【AI 回复】: {ai_reply}")

                            # Speaking
                            await websocket.send_json({"type": "status_change", "status": "speaking", "message": ai_reply})
                            
                            # Wait for Playback
                            logger.info(">>> 等待朗读...")
                            while True:
                                wait_msg = await websocket.receive()
                                if "text" in wait_msg:
                                    try:
                                        if json.loads(wait_msg["text"]).get("type") == "playback_finished":
                                            logger.info(">>> 朗读结束")
                                            break 
                                    except: pass
                                if wait_msg.get("type") == "websocket.disconnect":
                                    raise WebSocketDisconnect()

                            # Listening
                            is_user_turn = True
                            await websocket.send_json({"type": "status_change", "status": "listening", "message": "Listening..."})

                    except json.JSONDecodeError:
                        pass
                
                if message.get("type") == "websocket.disconnect":
                    break

        except WebSocketDisconnect:
            logger.info("前端断开连接")
        except Exception as e:
            logger.error(f"主循环异常: {e}")
        finally:
            try:
                if asr_client.ws: await asr_client.ws.close()
            except: pass

    # --- 任务 B: 接收 ASR 结果 (重点修复了这里) ---
    async def receive_asr_result():
        nonlocal current_full_text
        while True:
            try:
                # 【修复核心】如果 ASR 断开了（ws is None），我们就不去“收听”了
                # 我们就静静地睡着，直到 receive_from_frontend 把它重连上
                if asr_client.ws is None:
                    await asyncio.sleep(0.1) # 极短的休眠，等待重连
                    continue

                # 正常接收
                try:
                    async for text in asr_client.receive_result():
                        cleaned = re.sub(r'[\u4e00-\u9fa5]', '', text)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        current_full_text = cleaned
                        
                        if len(current_full_text) > last_committed_length:
                            live = current_full_text[last_committed_length:]
                            if is_user_turn:
                                await websocket.send_json({"type": "text", "content": live})
                except Exception as e:
                    # 如果 receive_result 内部报错（比如 Error Frame），说明连接真的断了
                    logger.warning(f"ASR 连接超时断开 (这是正常的): {e}")
                
                # 【关键】一旦跳出 async for 循环（说明断开了），必须把 ws 设为 None
                # 这样下一次循环就会进入上面的 sleep 等待模式，就不会刷屏了
                asr_client.ws = None
                
            except Exception as outer_e:
                logger.error(f"接收线程异常: {outer_e}")
                await asyncio.sleep(1.0)

    task1 = asyncio.create_task(receive_from_frontend())
    task2 = asyncio.create_task(receive_asr_result())
    done, pending = await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)
    for t in pending: t.cancel()