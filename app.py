from flask import Flask, render_template, request, jsonify
from utils import (
    download_youtube_video,
    extract_frames,
    describe_frames,
    summarize_with_gemma,
    transcribe_audio
)

app = Flask(__name__)

# Store video context for chat
video_context = {
    "summary": "",
    "transcript": "",
    "frame_descriptions": []
}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            youtube_url = is_valid_youtube_url(request.form["youtube_url"])
            
            if not youtube_url.startswith(("https://youtube.com", "https://www.youtube.com")):
                return jsonify({"error": "Invalid YouTube URL"}), 400
                
            video_path = download_youtube_video(youtube_url)
            # ... rest of your processing code ...
            return jsonify({"summary": summary})
            
        except Exception as e:
            return jsonify({
                "error": "Failed to process video",
                "details": str(e)
            }), 500
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    user_question = request.json["question"]
    context = f"""
        Video Summary: {video_context["summary"]}
        Transcript: {video_context["transcript"]}
        Frame Descriptions: {video_context["frame_descriptions"]}
    """
    
    # Query Ollama Gemma 3 with context
    response = describe_frames(context, prompt=user_question)  # Reuse Ollama call
    return jsonify({"answer": response})

def is_valid_youtube_url(url):
    return re.match(r'^(https?\:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$', url)
    
if __name__ == "__main__":
    app.run(debug=True)