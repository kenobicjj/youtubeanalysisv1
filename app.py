import re
from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import threading
import datetime
import faiss
import numpy as np
from utils import (
    download_youtube_video,
    extract_frames,
    describe_frames,
    summarize_with_gemma,
    transcribe_audio,
    embed_texts,
    retrieve_from_rag
)
import shutil
import os

app = Flask(__name__)

# Store video context for chat
video_context = {
    "summary": "",
    "transcript": "",
    "frame_descriptions": []
}

status_message = ""
terminate_flag = {"stop": False}
analysis_times = {"start": None, "end": None}
vectorstore = {"index": None, "texts": []}

def set_status(msg):
    global status_message
    status_message = msg
    print(msg)

def analyze_video(youtube_url):
    try:
        terminate_flag["stop"] = False
        analysis_times["start"] = datetime.datetime.now()
        analysis_times["end"] = None
        set_status("Downloading video...")
        if terminate_flag["stop"]: set_status("Terminated."); return
        video_path = download_youtube_video(youtube_url)
        set_status("Extracting frames...")
        if terminate_flag["stop"]: set_status("Terminated."); return
        frames = extract_frames(video_path)
        set_status("Transcribing audio...")
        if terminate_flag["stop"]: set_status("Terminated."); return
        transcript = transcribe_audio(video_path)
        set_status("Describing frames...")
        if terminate_flag["stop"]: set_status("Terminated."); return
        frame_descriptions = describe_frames(frames, set_status=set_status)
        set_status("Extracting audio features...")
        if terminate_flag["stop"]: set_status("Terminated."); return
        # Extract audio features (simple: use transcript segments as proxy, or use audio embedding model if available)
        # Here, we use transcript segments as audio context, but you could use a model like VGGish for real audio embeddings
        audio_segments = [f"Audio segment: {seg.strip()}" for seg in transcript.split('. ') if seg.strip()]
        set_status("Summarizing video...")
        if terminate_flag["stop"]: set_status("Terminated."); return
        summary = summarize_with_gemma(transcript)
        global video_context, vectorstore
        video_context["summary"] = summary
        video_context["transcript"] = transcript
        video_context["frame_descriptions"] = frame_descriptions
        # Build multimodal vectorstore: transcript segments, frame descriptions, audio segments
        texts = []
        transcript_segs = [seg.strip() for seg in transcript.split('. ') if seg.strip()]
        texts.extend(transcript_segs)
        texts.extend(frame_descriptions)
        texts.extend(audio_segments)
        embeddings = embed_texts(texts)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype=np.float32))
        vectorstore = {"index": index, "texts": texts}
        # Save RAG context and vectorstore
        import pickle
        with open("rag_context.pkl", "wb") as f:
            pickle.dump({"video_context": video_context, "vectorstore_texts": texts}, f)
        faiss.write_index(index, "rag_faiss.index")
        analysis_times["end"] = datetime.datetime.now()
        duration = (analysis_times["end"] - analysis_times["start"]).total_seconds()
        set_status(f"Analysis complete. Start: {analysis_times['start'].strftime('%Y-%m-%d %H:%M:%S')}, End: {analysis_times['end'].strftime('%Y-%m-%d %H:%M:%S')}, Duration: {duration:.1f} seconds.")
    except Exception as e:
        set_status(f"Error processing video: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        youtube_url = request.form["youtube_url"]
        if not is_valid_youtube_url(youtube_url):
            return jsonify({"error": "Invalid YouTube URL"}), 400
        set_status("Processing video, please wait...")
        # Start analysis in a background thread
        threading.Thread(target=analyze_video, args=(youtube_url,)).start()
        return jsonify({"status": "processing", "message": "Video is being analyzed. Please wait..."})
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    user_question = request.json["question"]
    global vectorstore
    # Load vectorstore if not in memory
    if vectorstore["index"] is None and os.path.exists("rag_faiss.index"):
        import pickle
        vectorstore["index"] = faiss.read_index("rag_faiss.index")
        with open("rag_context.pkl", "rb") as f:
            data = pickle.load(f)
            vectorstore["texts"] = data["vectorstore_texts"]
    if vectorstore["index"] is None or not vectorstore["texts"]:
        return jsonify({"answer": "No context available. Please analyze a video first."})
    # Embed question and search
    q_emb = embed_texts([user_question])
    D, I = vectorstore["index"].search(np.array(q_emb, dtype=np.float32), k=3)
    retrieved = [vectorstore["texts"][i] for i in I[0] if i < len(vectorstore["texts"])]
    context = "\n".join(retrieved)
    answer = summarize_with_gemma(f"Context: {context}\nQuestion: {user_question}")
    return jsonify({"answer": answer})

@app.route("/reset", methods=["POST"])
def reset():
    global video_context, vectorstore
    video_context = {
        "summary": "",
        "transcript": "",
        "frame_descriptions": []
    }
    vectorstore = {"index": None, "texts": []}
    # Optionally, clear frames directory and video file
    frames_dir = "frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    if os.path.exists("video.mp4"):
        os.remove("video.mp4")
    set_status("Session reset.")
    return jsonify({"success": True, "message": "Session reset."})

@app.route("/load", methods=["POST"])
def load_last_session():
    import pickle
    global video_context, vectorstore
    rag_path = "rag_context.pkl"
    if os.path.exists(rag_path):
        with open(rag_path, "rb") as f:
            data = pickle.load(f)
            video_context = data["video_context"]
            vectorstore["texts"] = data["vectorstore_texts"]
        if os.path.exists("rag_faiss.index"):
            vectorstore["index"] = faiss.read_index("rag_faiss.index")
        set_status("Last session loaded.")
        return jsonify({"success": True, "summary": video_context.get("summary", "")})
    else:
        video_context = {
            "summary": "",
            "transcript": "",
            "frame_descriptions": []
        }
        vectorstore = {"index": None, "texts": []}
        set_status("No saved session found. Ready for new analysis.")
        return jsonify({"success": False, "error": "No saved session found. Ready for new analysis."})

@app.route("/status")
def status():
    return jsonify({
        "status": status_message,
        "start": analysis_times["start"].strftime('%Y-%m-%d %H:%M:%S') if analysis_times["start"] else None,
        "end": analysis_times["end"].strftime('%Y-%m-%d %H:%M:%S') if analysis_times["end"] else None
    })

@app.route("/terminate", methods=["POST"])
def terminate():
    terminate_flag["stop"] = True
    set_status("Terminating analysis...")
    return jsonify({"success": True, "message": "Termination signal sent."})

def is_valid_youtube_url(url):
    return re.match(r'^(https?\:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$', url)
    
if __name__ == "__main__":
    app.run(debug=True)