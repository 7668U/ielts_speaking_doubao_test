import asyncio
import logging
import json
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from asr_service import VolcASRClient
from llm_service import DeepSeekClient

# ================= 配置区域 =================
ASR_APPID = "7407835878"
ASR_TOKEN = "Q3tgMiF7q_crzVMR0kPv8Jbk1JHF5kx7"
ASR_CLUSTER = "volc.bigasr.sauc.duration"
DEEPSEEK_API_KEY = "sk-e357919534ea4b098ec864c29d2166fc"

# 考试时间配置（秒）- 这些是硬性上限
EXAM_CONFIG = {
    "part1_duration": 300,      # Part 1 最长 5 分钟
    "part2_prep_duration": 60,  # 准备时间固定 1 分钟
    "part2_duration": 120,      # Part 2 最长 2 分钟
    "part3_duration": 300,      # Part 3 最长 5 分钟
}

# Part 2 话题卡
PART2_TOPIC = {
    "title": "Describe a place you would like to visit",
    "points": [
        "Where it is",
        "How you would go there",
        "What you would do there",
        "Why you would like to visit this place"
    ]
}

# 结构化输出的 System Prompt
IELTS_PROMPT = """You are a professional IELTS Speaking Examiner conducting a real IELTS Speaking test.

## Test Structure
- **Part 1** (4-5 min): Introduction & simple questions about familiar topics (home, work, studies, hobbies, etc.)
- **Part 2** (3-4 min): Long turn - candidate speaks for 1-2 minutes on a topic card
- **Part 3** (4-5 min): Discussion - deeper, abstract questions related to Part 2 topic

## Current Part 2 Topic Card
"Describe a place you would like to visit"
- Where it is
- How you would go there
- What you would do there
- Why you would like to visit this place

## Your Behavior Rules
1. **Part 1**: Ask 4-6 simple, personal questions about different topics. Start with name, then hometown/work/study, then interests.
2. **Part 2 Transition**: After sufficient Part 1 questions (usually 4-6 exchanges), YOU initiate the transition by saying something like "Thank you. Now I'd like to give you a topic..."
3. **Part 2**: Just listen. After candidate finishes, you may ask 1-2 brief follow-up questions.
4. **Part 3 Transition**: After Part 2, say "Now let's discuss some more general questions..."
5. **Part 3**: Ask abstract, analytical questions about travel, places, tourism, globalization, etc.

## CRITICAL: Response Format
You MUST respond in this exact JSON format. No other text allowed:

{
    "reply": "Your spoken response here",
    "current_part": 1,
    "should_transition_to_part2": false,
    "should_transition_to_part3": false
}

### Field Definitions:
- `reply`: What you say to the candidate (1-2 sentences, natural examiner speech)
- `current_part`: Which part you believe we're currently in (1, 2, or 3)
- `should_transition_to_part2`: Set TRUE only when you're ready to END Part 1 and START Part 2 preparation
- `should_transition_to_part3`: Set TRUE only when you're ready to END Part 2 and START Part 3

### Transition Examples:
When transitioning to Part 2:
{"reply": "Thank you. That's the end of Part 1. Now I'm going to give you a topic and I'd like you to talk about it for one to two minutes. Before you talk, you'll have one minute to prepare.", "current_part": 1, "should_transition_to_part2": true, "should_transition_to_part3": false}

When transitioning to Part 3:
{"reply": "Thank you. Now let's move on to Part 3. I'd like to discuss some questions related to travel and places.", "current_part": 2, "should_transition_to_part3": true, "should_transition_to_part2": false}

Remember: Keep responses concise and natural. Ask ONE question at a time. Do not use markdown formatting in your reply text."""


class ExamPhase(Enum):
    IDLE = "idle"
    PART1 = "part1"
    PART1_END = "part1_end"
    PART2_PREP = "part2_prep"
    PART2 = "part2"
    PART2_END = "part2_end"
    PART3 = "part3"
    EXAM_END = "exam_end"


@dataclass
class ExamState:
    phase: ExamPhase = ExamPhase.IDLE
    phase_start_time: Optional[float] = None
    is_accepting_audio: bool = False
    is_user_turn: bool = True
    transition_in_progress: bool = False
    ai_detected_part: int = 1  # AI 认为当前是哪个 Part


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
    logger.info(">>> WebSocket 已连接")

    llm_client = DeepSeekClient(api_key=DEEPSEEK_API_KEY, system_prompt=IELTS_PROMPT)
    asr_client = VolcASRClient(appid=ASR_APPID, token=ASR_TOKEN, cluster=ASR_CLUSTER)

    exam_state = ExamState()
    current_full_text = ""
    last_committed_length = 0
    shutdown_event = asyncio.Event()
    prep_countdown_task: Optional[asyncio.Task] = None

    def get_phase_duration(phase: ExamPhase) -> int:
        duration_map = {
            ExamPhase.PART1: EXAM_CONFIG["part1_duration"],
            ExamPhase.PART2_PREP: EXAM_CONFIG["part2_prep_duration"],
            ExamPhase.PART2: EXAM_CONFIG["part2_duration"],
            ExamPhase.PART3: EXAM_CONFIG["part3_duration"],
        }
        return duration_map.get(phase, 0)

    def get_remaining_time() -> int:
        if exam_state.phase_start_time is None:
            return get_phase_duration(exam_state.phase)
        elapsed = asyncio.get_event_loop().time() - exam_state.phase_start_time
        remaining = get_phase_duration(exam_state.phase) - elapsed
        return max(0, int(remaining))

    async def safe_send_json(data: dict):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.warning(f"发送失败: {e}")
            shutdown_event.set()

    async def send_exam_state(prep_remaining: int = None):
        phase_display = {
            ExamPhase.IDLE: "Ready",
            ExamPhase.PART1: "Part 1: Introduction",
            ExamPhase.PART1_END: "Transitioning...",
            ExamPhase.PART2_PREP: "Part 2: Preparation",
            ExamPhase.PART2: "Part 2: Your Turn",
            ExamPhase.PART2_END: "Transitioning...",
            ExamPhase.PART3: "Part 3: Discussion",
            ExamPhase.EXAM_END: "Exam Completed",
        }
        
        state_data = {
            "type": "exam_state",
            "phase": exam_state.phase.value,
            "phase_display": phase_display.get(exam_state.phase, ""),
            "remaining_seconds": get_remaining_time(),
            "total_seconds": get_phase_duration(exam_state.phase),
            "accepting_audio": exam_state.is_accepting_audio,
            "ai_detected_part": exam_state.ai_detected_part,
        }
        
        if exam_state.phase in [ExamPhase.PART2_PREP, ExamPhase.PART2]:
            state_data["part2_topic"] = PART2_TOPIC
        
        if exam_state.phase == ExamPhase.PART2_PREP and prep_remaining is not None:
            state_data["prep_countdown"] = prep_remaining
            state_data["prep_total"] = EXAM_CONFIG["part2_prep_duration"]
        
        await safe_send_json(state_data)

    async def wait_for_playback():
        logger.info(">>> 等待 TTS 播放完成...")
        while not shutdown_event.is_set():
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=60.0)
                if "text" in msg:
                    try:
                        data = json.loads(msg["text"])
                        if data.get("type") == "playback_finished":
                            logger.info(">>> TTS 播放完成")
                            return True
                    except json.JSONDecodeError:
                        pass
                if msg.get("type") == "websocket.disconnect":
                    shutdown_event.set()
                    return False
            except asyncio.TimeoutError:
                logger.warning("等待播放超时")
                return False
        return False

    async def play_system_message(message: str):
        exam_state.is_accepting_audio = False
        exam_state.is_user_turn = False
        await safe_send_json({
            "type": "system_speak",
            "content": message
        })
        await wait_for_playback()

    def parse_ai_response(response: str) -> dict:
        """解析 AI 的结构化 JSON 响应"""
        default = {
            "reply": response,
            "current_part": exam_state.ai_detected_part,
            "should_transition_to_part2": False,
            "should_transition_to_part3": False
        }
        
        try:
            # 尝试提取 JSON（可能被 markdown 包裹）
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "reply": data.get("reply", response),
                    "current_part": data.get("current_part", exam_state.ai_detected_part),
                    "should_transition_to_part2": data.get("should_transition_to_part2", False),
                    "should_transition_to_part3": data.get("should_transition_to_part3", False)
                }
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"JSON 解析失败: {e}, 原始响应: {response[:200]}")
        
        return default

    async def start_part2_prep():
        """启动 Part 2 准备阶段"""
        nonlocal prep_countdown_task
        
        exam_state.phase = ExamPhase.PART2_PREP
        exam_state.phase_start_time = asyncio.get_event_loop().time()
        exam_state.is_accepting_audio = False
        exam_state.is_user_turn = False
        exam_state.ai_detected_part = 2
        
        await send_exam_state(prep_remaining=EXAM_CONFIG["part2_prep_duration"])
        
        # 启动倒计时
        prep_countdown_task = asyncio.create_task(run_part2_prep_countdown())

    async def run_part2_prep_countdown():
        """Part 2 准备阶段倒计时"""
        logger.info(">>> Part 2 准备阶段开始")
        duration = EXAM_CONFIG["part2_prep_duration"]
        
        for remaining in range(duration, 0, -1):
            if shutdown_event.is_set():
                return
            await send_exam_state(prep_remaining=remaining)
            await asyncio.sleep(1)
        
        await send_exam_state(prep_remaining=0)
        
        if shutdown_event.is_set():
            return
            
        logger.info(">>> Part 2 准备时间结束，播放提示语")
        
        # 发送语音提示（不等待完成，避免和主循环冲突）
        await safe_send_json({
            "type": "system_speak", 
            "content": "Alright, your preparation time is over. Please begin your talk now."
        })
        
        # 等待语音播放完成（估算时间，约4秒）
        await asyncio.sleep(4)
        
        if shutdown_event.is_set():
            return
        
        # 切换到 Part 2 陈述阶段
        logger.info(">>> 进入 Part 2 陈述阶段")
        exam_state.phase = ExamPhase.PART2
        exam_state.phase_start_time = asyncio.get_event_loop().time()
        exam_state.is_accepting_audio = True
        exam_state.is_user_turn = True
        
        await send_exam_state()
        await safe_send_json({
            "type": "status_change",
            "status": "listening",
            "message": "Please begin speaking..."
        })

    async def start_part3():
        """启动 Part 3"""
        exam_state.phase = ExamPhase.PART3
        exam_state.phase_start_time = asyncio.get_event_loop().time()
        exam_state.ai_detected_part = 3
        await send_exam_state()
        
        # 让 AI 开始 Part 3 提问
        ai_reply = await llm_client.get_response(
            "[SYSTEM: Part 2 has just ended. Begin Part 3 now with your first discussion question about travel/places. Remember to respond in JSON format.]"
        )
        parsed = parse_ai_response(ai_reply)
        logger.info(f"【AI Part 3 开场】: {parsed['reply']}")
        
        await safe_send_json({
            "type": "status_change",
            "status": "speaking",
            "message": parsed["reply"]
        })
        
        # 估算等待时间（根据文字长度）
        wait_time = max(3, len(parsed["reply"]) / 15)  # 大约每秒15个字符
        await asyncio.sleep(wait_time)
        
        exam_state.is_accepting_audio = True
        exam_state.is_user_turn = True
        await send_exam_state()
        await safe_send_json({
            "type": "status_change",
            "status": "listening",
            "message": "Listening..."
        })



    async def handle_phase_timeout():
        """处理硬性时间限制"""
        if exam_state.transition_in_progress:
            return
            
        exam_state.transition_in_progress = True
        logger.info(f">>> 阶段 {exam_state.phase.value} 时间到! (硬性限制)")

        try:
            if exam_state.phase == ExamPhase.PART1:
                exam_state.phase = ExamPhase.PART1_END
                exam_state.is_accepting_audio = False
                await send_exam_state()
                
                # 发送语音（不等待）
                await safe_send_json({
                    "type": "system_speak",
                    "content": "Thank you. That brings us to the end of Part 1. Now I'm going to give you a topic and I'd like you to talk about it for one to two minutes. You have one minute to prepare. Here is your topic."
                })
                await asyncio.sleep(8)  # 等待语音播放
                
                await start_part2_prep()
                
            elif exam_state.phase == ExamPhase.PART2:
                exam_state.phase = ExamPhase.PART2_END
                exam_state.is_accepting_audio = False
                await send_exam_state()
                
                # 发送语音（不等待）
                await safe_send_json({
                    "type": "system_speak",
                    "content": "Thank you. That's the end of your long turn. Now let's move on to Part 3."
                })
                await asyncio.sleep(5)  # 等待语音播放
                
                await start_part3()
                
            elif exam_state.phase == ExamPhase.PART3:
                exam_state.phase = ExamPhase.EXAM_END
                exam_state.is_accepting_audio = False
                await send_exam_state()
                
                # 发送语音（不等待）
                await safe_send_json({
                    "type": "system_speak",
                    "content": "Thank you very much. That is the end of the speaking test. Thank you for your participation. Goodbye!"
                })
                await asyncio.sleep(5)
                
        finally:
            exam_state.transition_in_progress = False

    async def handle_ai_transition(parsed_response: dict):
        """处理 AI 主动触发的过渡"""
        if exam_state.transition_in_progress:
            return False
        
        # AI 建议过渡到 Part 2
        if parsed_response.get("should_transition_to_part2") and exam_state.phase == ExamPhase.PART1:
            logger.info(">>> AI 触发 Part 1 → Part 2 过渡")
            exam_state.transition_in_progress = True
            
            try:
                exam_state.phase = ExamPhase.PART1_END
                exam_state.is_accepting_audio = False
                await send_exam_state()
                
                # AI 的过渡语已经播放了，直接进入准备阶段
                await start_part2_prep()
                return True
            finally:
                exam_state.transition_in_progress = False
        
        # AI 建议过渡到 Part 3
        elif parsed_response.get("should_transition_to_part3") and exam_state.phase == ExamPhase.PART2:
            logger.info(">>> AI 触发 Part 2 → Part 3 过渡")
            exam_state.transition_in_progress = True
            
            try:
                exam_state.phase = ExamPhase.PART2_END
                exam_state.is_accepting_audio = False
                await send_exam_state()
                
                await start_part3()
                return True
            finally:
                exam_state.transition_in_progress = False
        
        return False

    # ASR 连接
    async def ensure_asr_connected():
        try:
            if asr_client.ws is not None:
                return True
            logger.info(">>> 正在建立 ASR 连接...")
            connected = await asr_client.connect()
            if connected:
                await asr_client.send_full_client_request()
                logger.info(">>> ASR 服务已就绪")
            return connected
        except Exception as e:
            logger.error(f"ASR 连接失败: {e}")
            return False

    await ensure_asr_connected()

    # 任务1: 阶段计时器
    async def phase_timer():
        while not shutdown_event.is_set():
            try:
                await asyncio.sleep(1)
                
                if shutdown_event.is_set():
                    break
                
                # Part 2 Prep 由专门任务处理
                if exam_state.phase == ExamPhase.PART2_PREP:
                    continue
                
                if exam_state.phase in [ExamPhase.PART1, ExamPhase.PART2, ExamPhase.PART3]:
                    if exam_state.phase_start_time is not None:
                        remaining = get_remaining_time()
                        await send_exam_state()
                        
                        # 硬性时间限制
                        if remaining <= 0 and not exam_state.transition_in_progress:
                            await handle_phase_timeout()
                            
            except Exception as e:
                logger.error(f"计时器异常: {e}")
                if "disconnect" in str(e).lower():
                    break

    # 任务2: 接收前端消息
    async def receive_from_frontend():
        nonlocal current_full_text, last_committed_length

        try:
            while not shutdown_event.is_set():
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if "bytes" in message and message["bytes"]:
                    if exam_state.phase == ExamPhase.IDLE:
                        logger.info(">>> 开始 Part 1")
                        exam_state.phase = ExamPhase.PART1
                        exam_state.phase_start_time = asyncio.get_event_loop().time()
                        exam_state.is_accepting_audio = True
                        exam_state.is_user_turn = True
                        exam_state.ai_detected_part = 1
                        await send_exam_state()

                    if exam_state.is_accepting_audio and exam_state.is_user_turn:
                        try:
                            if asr_client.ws is None:
                                if await ensure_asr_connected():
                                    current_full_text = ""
                                    last_committed_length = 0
                            
                            if asr_client.ws is not None:
                                await asr_client.send_audio_chunk(message["bytes"], is_last=False)
                        except Exception as e:
                            logger.warning(f"发送音频失败: {e}")
                            asr_client.ws = None

                elif "text" in message and message["text"]:
                    try:
                        data = json.loads(message["text"])

                        if data.get("type") == "done_speaking":
                            if not exam_state.is_user_turn or not exam_state.is_accepting_audio:
                                continue
                            
                            if exam_state.transition_in_progress:
                                continue

                            final_text = current_full_text[last_committed_length:].strip()
                            last_committed_length = len(current_full_text)

                            if not final_text or final_text == "...":
                                await safe_send_json({
                                    "type": "status_change",
                                    "status": "listening",
                                    "message": "Listening..."
                                })
                                continue

                            logger.info(f"【用户】: {final_text}")
                            exam_state.is_user_turn = False

                            await safe_send_json({
                                "type": "status_change",
                                "status": "processing",
                                "message": f"You: {final_text}"
                            })

                            # Part 2 是用户独白
                            if exam_state.phase == ExamPhase.PART2:
                                exam_state.is_user_turn = True
                                await safe_send_json({
                                    "type": "status_change",
                                    "status": "listening",
                                    "message": "Continue speaking..."
                                })
                                continue

                            # Part 1 或 Part 3，需要 AI 回复
                            ai_reply = await llm_client.get_response(final_text)
                            parsed = parse_ai_response(ai_reply)
                            
                            # 更新 AI 检测到的 Part
                            exam_state.ai_detected_part = parsed.get("current_part", exam_state.ai_detected_part)
                            logger.info(f"【AI】(Part {exam_state.ai_detected_part}): {parsed['reply']}")

                            await safe_send_json({
                                "type": "status_change",
                                "status": "speaking",
                                "message": parsed["reply"]
                            })

                            await wait_for_playback()
                            
                            # 检查 AI 是否触发过渡
                            transitioned = await handle_ai_transition(parsed)
                            
                            # 如果没有过渡，继续监听
                            if not transitioned and not exam_state.transition_in_progress:
                                exam_state.is_user_turn = True
                                await safe_send_json({
                                    "type": "status_change",
                                    "status": "listening",
                                    "message": "Listening..."
                                })

                    except json.JSONDecodeError:
                        pass

                if message.get("type") == "websocket.disconnect":
                    shutdown_event.set()
                    break

        except WebSocketDisconnect:
            logger.info("前端断开连接")
        except Exception as e:
            logger.error(f"主循环异常: {e}")
        finally:
            shutdown_event.set()

    # 任务3: 接收 ASR 结果
    async def receive_asr_result():
        nonlocal current_full_text
        
        while not shutdown_event.is_set():
            try:
                if asr_client.ws is None:
                    await asyncio.sleep(0.2)
                    continue

                try:
                    async for text in asr_client.receive_result():
                        if shutdown_event.is_set():
                            break
                            
                        cleaned = re.sub(r'[\u4e00-\u9fa5]', '', text)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        current_full_text = cleaned

                        if len(current_full_text) > last_committed_length:
                            live = current_full_text[last_committed_length:]
                            if exam_state.is_user_turn and exam_state.is_accepting_audio:
                                await safe_send_json({"type": "text", "content": live})
                                
                except Exception as e:
                    if not shutdown_event.is_set():
                        logger.warning(f"ASR 接收中断: {e}")

                asr_client.ws = None

            except Exception as e:
                if not shutdown_event.is_set():
                    logger.error(f"ASR 任务异常: {e}")
                await asyncio.sleep(0.5)

    # 启动任务
    try:
        tasks = [
            asyncio.create_task(phase_timer(), name="timer"),
            asyncio.create_task(receive_from_frontend(), name="frontend"),
            asyncio.create_task(receive_asr_result(), name="asr"),
        ]
        
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        shutdown_event.set()
        
        if prep_countdown_task and not prep_countdown_task.done():
            prep_countdown_task.cancel()
        
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
    except Exception as e:
        logger.error(f"主任务异常: {e}")
    finally:
        try:
            if asr_client.ws:
                await asr_client.ws.close()
        except:
            pass
        logger.info(">>> Session 结束")