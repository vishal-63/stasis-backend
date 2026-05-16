import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ReelRequest(BaseModel):
    url: str

def download_audio(url: str):
    # Generates a unique filename to avoid collisions
    filename = f"audio_{uuid.uuid4()}"
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'cookiefile': 'cookies.txt',
        'outtmpl': filename,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return f"{filename}.mp3"

def get_reel_data(url: str):
    filename = f"video_{uuid.uuid4()}"
    # Change format to 'mp4' to allow for OCR later
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': filename,
        'cookiefile': 'cookies.txt',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        description = info.get('description', '')

    audio_path = download_audio(url)
    
    return f"{filename}.mp4", description, audio_path

def filter_transcript(transcript_text, description_text):
    if len(transcript_text.split()) < 8:
        return ""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are a Noise Filter. Compare the Audio Transcript to the Video Description. If the Audio Transcript consists of song lyrics, background music descriptions, or content entirely unrelated to the factual information in the Description, return 'IRRELEVANT'. Otherwise, return the original transcript."
                )
            },
            {
                "role": "user", 
                "content": f"Description: {description_text}\nAudio: {transcript_text}"
            }
        ],
        temperature=0
    )
    
    result = response.choices[0].message.content
    return "" if "IRRELEVANT" in result else transcript_text

@app.post("/process-reel")
async def process_reel(request: ReelRequest):
    audio_path = None
    try:
        # 2. Get Video, Description & Audio from the Reel URL
        video_path, description, audio_path = get_reel_data(request.url)
        
        # 2. Transcribe using Whisper
        with open(audio_path, "rb") as audio_file:
            raw_transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )

        # 3. Filter out irrelevant transcripts (like music or chanting)
        sanitized_transcript = filter_transcript(raw_transcript.text, description)


        # 4. TODO: Add a feature to Run OCR or Audio Transcription

        # 5. Use GPT-4o-mini to extract structured data
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a 'Zero-Fluff' Information Architect. "
                        "Your task is to strip away all social media performance and extract the core utility of the video. "
                        "STRICT GUIDELINES: "
                        "1. IDENTIFY THE TYPE: Automatically detect if the content is a 'List', 'Process/Tutorial', 'Review', or 'News/Update'. "
                        "2. STRUCTURE BY TYPE: "
                        "   - If a List: Provide a bulleted list of items with their specific details. "
                        "   - If a Process: Provide numbered chronological steps. "
                        "   - If a Review: Provide Pros, Cons, and the Final Verdict. "
                        "3. PRESERVE SPECIFICS: Never generalize. If a video mentions 'The $500 Sony ZV-E10 camera', do not write 'a camera'. Write the full name and price. "
                        "4. NO INTRODUCTIONS: Do not say 'This video is about...'. Start with the primary subject as a Header. "
                        "5. VISUAL CONTEXT: If the transcript is sparse but the description is rich (or vice versa), prioritize the denser source. "
                        "If the Audio Transcript contains poetic, rhythmic, or lyrical sentences that do not align with the factual keywords in the Description, treat the Audio as 'Background Music' and DISCARD it. Only use the Audio if it provides spoken instructions, facts, or commentary that supports the Description."
                        "Output your response in JSON format. "
                        "Include a 'summary' string, and a 'tags' array. "
                        "Tags should be lowercase, one-word, and highly relevant categories. "
                        "Example tags: #recipe, #productivity, #tech, #travel. "
                        "Do not use more than 5 tags."
                    )
                },
                {
                    "role": "user", 
                    "content": f"INPUT DATA:\nDescription/Caption: {description}\nAudio Transcript: {sanitized_transcript if sanitized_transcript else 'N/A'}"
                }
            ],
            temperature=0
        )
        
        data = json.loads(response.choices[0].message.content)
        summary = data.get("summary", "")
        tags = data.get("tags", [])
        return {"summary": summary, "transcript": sanitized_transcript if sanitized_transcript else 'N/A', "description": description, "tags": tags}

    except Exception as e:
        print(f"Error processing reel: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup: Delete the audio file so we don't fill up the server
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        if video_path and os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)