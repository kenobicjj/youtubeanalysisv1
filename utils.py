import cv2
import os

os.environ["PATH"] = r"C:\Users\JCHIN\ffmpeg\bin;" + os.environ["PATH"]
import whisper
import traceback
import yt_dlp
import ollama
#RAG
from sentence_transformers import SentenceTransformer
import numpy as np
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize models
whisper_model = whisper.load_model("base")
ollama_client = ollama.Client(host='http://localhost:11434')

def download_youtube_video(url, filename="video.mp4"):
    try:
        print(f"Attempting to download: {url}")
        ydl_opts = {
            'outtmpl': filename,
            'format': 'mp4/bestaudio/best',
            'merge_output_format': 'mp4'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return filename
    except Exception as e:
        print(f"Download failed: {str(e)}")
        import traceback; traceback.print_exc()
        raise

def extract_frames(video_path, output_folder="frames", frame_interval=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % int(fps * frame_interval) == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        frame_count += 1
    cap.release()
    return frames

def describe_frames(frame_paths, set_status=None, prompt="Describe this image in detail:"):
    descriptions = []
    for idx, frame in enumerate(frame_paths, 1):
        response = ollama_client.generate(
            model="gemma3",
            prompt=f"{prompt}\n{frame}",
            stream=False
        )
        descriptions.append(response["response"])
        if set_status:
            set_status(f"Described {idx}/{len(frame_paths)} frames...")
    return descriptions  # Always return a list

def summarize_with_gemma(text):
    response = ollama_client.generate(
        model="gemma3",
        prompt=f"Summarize this in one paragraph:\n{text}",
        stream=False
    )
    return response["response"]

def transcribe_audio(video_path):
    result = whisper_model.transcribe(video_path)
    return result["text"]

#RAG defs
def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True)

def find_most_similar(query, texts, embeddings):
    query_emb = embedder.encode([query], convert_to_numpy=True)[0]
    sims = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    idx = np.argmax(sims)
    return texts[idx]

def retrieve_from_rag(query, texts, embeddings, top_k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)[0]
    sims = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return "\n".join([texts[i] for i in top_indices])