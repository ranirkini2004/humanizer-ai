from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

# Import our AI logic
from humanizer import humanize_text

app = FastAPI(title="AI Text Humanizer API")

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request and response models
class HumanizeRequest(BaseModel):
    text: str
    style: str = "Casual"  # Default style


class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    style_applied: str


# API Endpoints
@app.get("/api/welcome")
def read_root():
    return {"message": "Welcome to the Humanize AI Text API! Send a POST request to /humanize."}


@app.post("/humanize", response_model=HumanizeResponse)
def humanize_endpoint(request: HumanizeRequest):
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Call the AI model
        rewritten_text = humanize_text(request.text, request.style)

        return HumanizeResponse(
            original_text=request.text,
            humanized_text=rewritten_text,
            style_applied=request.style
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount Frontend
# We check if the folder exists to prevent crash if running from wrong directory
if os.path.isdir("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    # Serve the index.html on the root URL
    return FileResponse("frontend/index.html")

# Run with: uvicorn main:app --reload