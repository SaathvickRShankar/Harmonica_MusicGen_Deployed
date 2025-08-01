# app.py (Final Version)
import os
import uuid
import torch
import torchaudio
import numpy as np
import librosa
import traceback
import streamlit as st
from dotenv import load_dotenv

from typing import Annotated, List, Literal
from typing_extensions import TypedDict

# LangChain & LangGraph specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END

# Model loading imports
from audiocraft.models import MusicGen
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import torch.nn.functional as F

# --- 1. SETUP & CONFIGURATION ---

st.set_page_config(
    page_title="Melody AI - Your Music Companion",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main app styling */
    .main > div {
        padding-top: 2rem;
    }
    .main-header {
        text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px; margin-bottom: 2rem; color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .main-header h1 { font-size: 3rem; margin: 0; font-weight: 700; }
    .main-header p { font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.9; }
    /* Welcome card styling */
    .welcome-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2rem;
        border-radius: 20px; text-align: center; margin: 2rem 0; color: white;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    .welcome-card h2 { font-size: 2.5rem; margin-bottom: 1rem; font-weight: 600; }
    .welcome-card p { font-size: 1.1rem; margin-bottom: 2rem; opacity: 0.95; }
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
        border: none; border-radius: 25px; padding: 0.75rem 1.5rem; font-weight: 600;
        font-size: 0.9rem; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px; padding: 1rem; margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    /* Sidebar styling */
    .css-1d391kg { background: linear-gradient(180deg, #f8f9ff 0%, #e8efff 100%); }
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #667eea; border-radius: 15px; padding: 2rem;
        text-align: center; background: rgba(102, 126, 234, 0.05); transition: all 0.3s ease;
    }
    .stFileUploader:hover { border-color: #764ba2; background: rgba(102, 126, 234, 0.1); }
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;
        padding: 0.5rem 1rem; border-radius: 20px; display: inline-block;
        font-weight: 600; margin: 0.5rem 0;
    }
    .status-processing {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white;
        padding: 0.5rem 1rem; border-radius: 20px; display: inline-block;
        font-weight: 600; margin: 0.5rem 0; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

load_dotenv()

class MusicState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda left, right: left + right]
    selected_mode: str
    detected_emotion: str
    generated_music_path: str
    audio_file: str
    error: str | None
    finished: bool

CHATBOT_SYSTEM_PROMPT = SystemMessage(content=(
    "You are a friendly assistant named Melody. You can chat with the user about anything. "
    "You can also generate music based on emotions the user expresses (like 'I feel happy' or 'play something sad') or from an audio file they upload. "
    "If they ask for music, confirm the emotion you detected before proceeding. "
    "When music generation is successful, inform the user that the file has been created. The UI will handle displaying it. "
    "If an error occurs, apologize and explain the issue briefly."
))

# --- 2. MODEL AND GRAPH LOADING (CACHED) ---
@st.cache_resource(show_spinner="🎵 Loading AI models... Creating your musical experience...")
def load_models_and_graph():
    # ... (This entire section is unchanged and correct)
    # Load LLM
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("🔴 GOOGLE_API_KEY not found. Make sure you have created a .env file with your key.")
            st.stop()
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7, google_api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini LLM: {e}")
        llm = None
    # Load Music Generation Model
    try:
        musicgen = MusicGen.get_pretrained("facebook/musicgen-small")
    except Exception as e:
        st.error(f"❌ Failed to load MusicGen model: {e}")
        musicgen = None
    # Load Text Emotion Model
    try:
        text_emotion_model = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')
    except Exception as e:
        st.error(f"❌ Failed to load text emotion model: {e}")
        text_emotion_model = None
    # Load Speech Emotion Model
    try:
        hf_model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        speech_emotion_model = AutoModelForAudioClassification.from_pretrained(hf_model_id)
        speech_feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_id)
        speech_id2label = speech_emotion_model.config.id2label
    except Exception as e:
        st.error(f"❌ Failed to load speech emotion model: {e}")
        speech_emotion_model, speech_feature_extractor, speech_id2label = None, None, None
    # GRAPH DEFINITION
    def chatbot_node(state: MusicState) -> dict:
        if not llm: return {"messages": [AIMessage(content="My chat capability is offline.")]}
        llm_input_messages = [CHATBOT_SYSTEM_PROMPT] + state["messages"]
        try:
            response = llm.invoke(llm_input_messages)
            return {"messages": [response]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Sorry, I had trouble responding: {e}")]}
    def supervisor_node_logic(state: MusicState) -> dict:
        last_message_content = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_message_content = msg.content.strip()
                break
        potential_path = last_message_content.strip('\'"')
        if os.path.exists(potential_path) and any(potential_path.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac', '.ogg']):
            return {"selected_mode": "speech", "audio_file": potential_path, "error": None}
        if any(kw in last_message_content.lower() for kw in ["feel", "feeling", "sound like", "music for", "generate", "play", "happy", "sad", "angry", "excited"]):
            return {"selected_mode": "text", "error": None}
        return {"selected_mode": "chat", "error": None}
    def text_emotion_agent(state: MusicState) -> dict:
        if not text_emotion_model: return {"error": "Text emotion model not loaded."}
        last_human_message = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
        try:
            emotion_result = text_emotion_model(last_human_message)
            return {"detected_emotion": emotion_result[0]['label']}
        except Exception as e:
            return {"error": f"Failed to detect text emotion: {e}"}
    def speech_agent(state: MusicState) -> dict:
        if not speech_emotion_model: return {"error": "Speech emotion model is not loaded."}
        audio_path = state.get("audio_file", "").strip()
        if not audio_path or not os.path.exists(audio_path):
            return {"error": "Audio file path is missing or invalid."}
        try:
            sampling_rate = speech_feature_extractor.sampling_rate
            audio_array, _ = librosa.load(audio_path, sr=sampling_rate)
            inputs = speech_feature_extractor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = speech_emotion_model(**inputs).logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            return {"detected_emotion": speech_id2label[predicted_id]}
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Failed to process audio file: {e}"}
    def musicgen_node(state: MusicState) -> dict:
        if not musicgen: return {"error": "MusicGen model is not loaded."}
        emotion = state.get("detected_emotion", "neutral")
        prompt = f"Instrumental music expressing the feeling of {emotion}, cinematic, high quality"
        try:
            musicgen.set_generation_params(duration=8)
            generated_tensor = musicgen.generate([prompt], progress=False)
            output_dir = "generated_music"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{emotion.replace(' ', '_')}_{uuid.uuid4()}.wav"
            audio_path = os.path.join(output_dir, filename)
            torchaudio.save(audio_path, generated_tensor[0].cpu(), musicgen.sample_rate)
            confirmation_msg = f"🎵 Perfect! I've composed a beautiful {emotion} track just for you!"
            return {"generated_music_path": audio_path, "messages": [AIMessage(content=confirmation_msg, additional_kwargs={"music_path": audio_path})]}
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Music generation failed: {e}"}
    def handle_error(state: MusicState) -> dict:
        error = state.get("error", "Unknown error.")
        return {"messages": [AIMessage(content=f"😔 Sorry, something went wrong: {error}")]}
    def supervisor_branching_logic(state: MusicState):
        if state.get("error"): return "handle_error"
        mode = state.get("selected_mode")
        if mode == "text": return "text_agent"
        if mode == "speech": return "speech_agent"
        return "chatbot"
    def route_after_agent(state: MusicState):
        return "handle_error" if state.get("error") else "music_gen"
    def route_after_musicgen(state: MusicState):
        return "handle_error" if state.get("error") else END
    graph_builder = StateGraph(MusicState)
    graph_builder.add_node("supervisor", supervisor_node_logic)
    graph_builder.add_node("text_agent", text_emotion_agent)
    graph_builder.add_node("speech_agent", speech_agent)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("music_gen", musicgen_node)
    graph_builder.add_node("handle_error", handle_error)
    graph_builder.set_entry_point("supervisor")
    graph_builder.add_conditional_edges("supervisor", supervisor_branching_logic)
    graph_builder.add_conditional_edges("text_agent", route_after_agent)
    graph_builder.add_conditional_edges("speech_agent", route_after_agent)
    graph_builder.add_conditional_edges("music_gen", route_after_musicgen)
    graph_builder.add_edge("chatbot", END)
    graph_builder.add_edge("handle_error", END)
    return graph_builder.compile()

# --- 3. STREAMLIT UI AND INTERACTION LOGIC ---
music_graph = load_models_and_graph()
if not music_graph:
    st.warning("⚠️ Graph could not be compiled. The app cannot run.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

def handle_file_upload():
    if st.session_state.audio_uploader:
        uploaded_file = st.session_state.audio_uploader
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.file_to_process = temp_path

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;"><h2 style="color: white; margin: 0;">🎤 Audio Upload</h2><p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Transform your voice into music</p></div>""", unsafe_allow_html=True)
    st.file_uploader("🎵 Drop your audio file here", type=['wav', 'mp3', 'flac', 'ogg'], key="audio_uploader", on_change=handle_file_upload, help="Upload a WAV, MP3, FLAC, or OGG file to analyze emotions and generate matching music")
    st.markdown("""<div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1rem; border-radius: 15px; margin-top: 1rem;"><p style="margin: 0; text-align: center; font-weight: 600;">✨ Upload complete? The magic happens automatically!</p></div>""", unsafe_allow_html=True)
    st.markdown("### 🌟 Features")
    features = [("🎭", "Emotion Detection", "Analyze mood from text or speech"), ("🎼", "AI Music Generation", "Create unique compositions"), ("💬", "Smart Chat", "Natural conversation about music"), ("🎨", "Multiple Styles", "Various musical genres and moods")]
    for icon, title, desc in features:
        st.markdown(f"""<div style="display: flex; align-items: center; margin: 1rem 0; padding: 0.5rem; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);"><span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span><div><strong>{title}</strong><br><small style="color: #666;">{desc}</small></div></div>""", unsafe_allow_html=True)

# Main Content Area
if not st.session_state.messages:
    st.markdown("""<div class="main-header"><h1>🎵 Melody AI</h1><p>Your Personal AI Music Composer & Emotional Companion</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="welcome-card"><h2>Welcome to Your Musical Journey! 🎶</h2><p>I'm Melody, and I'm here to create the perfect soundtrack for your emotions. Whether you're feeling happy, sad, excited, or contemplative, I'll compose music that resonates with your soul.</p></div>""", unsafe_allow_html=True)
    st.markdown("### 🎯 Quick Start - Try These Examples:")
    col1, col2, col3 = st.columns(3)
    prompts = {
        "happy": ("🎉 Generate Happy Music", "I'm feeling really happy and energetic today! Generate some upbeat music"),
        "calm": ("🧘 Create Calm Music", "I need something calm and peaceful to relax"),
        "emotional": ("🎭 Make Emotional Music", "Create something sad and cinematic, like a movie soundtrack")
    }
    # ** THE FIX IS HERE **
    # Correctly add the prompt to the message list when a button is clicked.
    if st.button(prompts["happy"][0], use_container_width=True, key="happy"):
        st.session_state.messages.append(HumanMessage(content=prompts["happy"][1]))
        st.rerun()
    if st.button(prompts["calm"][0], use_container_width=True, key="calm"):
        st.session_state.messages.append(HumanMessage(content=prompts["calm"][1]))
        st.rerun()
    if st.button(prompts["emotional"][0], use_container_width=True, key="emotional"):
        st.session_state.messages.append(HumanMessage(content=prompts["emotional"][1]))
        st.rerun()
else:
    st.markdown("""<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1rem;"><h2 style="color: white; margin: 0;">🎵 Melody AI</h2></div>""", unsafe_allow_html=True)

# Chat messages with enhanced styling
for message in st.session_state.messages:
    avatar = "🎵" if isinstance(message, AIMessage) else "👤"
    with st.chat_message(message.type, avatar=avatar):
        if isinstance(message, HumanMessage) and message.content.startswith("temp_audio"):
            st.markdown('<div class="status-processing">🎤 Analyzing your audio file...</div>', unsafe_allow_html=True)
        else:
            st.markdown(message.content)
        if isinstance(message, AIMessage) and "music_path" in message.additional_kwargs:
            music_path = message.additional_kwargs["music_path"]
            if os.path.exists(music_path):
                st.markdown('<div class="status-success">🎼 Your music is ready!</div>', unsafe_allow_html=True)
                st.audio(music_path)
                with open(music_path, "rb") as file:
                    st.download_button(label="💾 Download Your Music", data=file.read(), file_name=os.path.basename(music_path), mime="audio/wav", use_container_width=True)

# Handle file processing
if st.session_state.get("file_to_process"):
    file_path = st.session_state.file_to_process
    st.session_state.messages.append(HumanMessage(content=file_path))
    del st.session_state.file_to_process
    st.rerun()

# Chat input
if prompt := st.chat_input("🎵 Describe your mood, ask for music, or just say hello...", key="prompt"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.rerun()

# Agent processing
if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
    with st.chat_message("ai", avatar="🎵"):
        with st.spinner("🎼 Melody is composing your perfect soundtrack..."):
            current_state = MusicState(messages=st.session_state.messages, selected_mode="", detected_emotion="", generated_music_path="", audio_file="", error=None, finished=False)
            final_state = music_graph.invoke(current_state, {"recursion_limit": 20})
            new_ai_message = final_state['messages'][-1]
            st.session_state.messages.append(new_ai_message)
    st.rerun()