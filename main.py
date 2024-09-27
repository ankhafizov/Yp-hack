from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import yaml
import urllib.request
from duplicates_db import check_video_duplicate
import uvicorn


app = FastAPI()

# Pydantic model for the request
class VideoLinkRequest(BaseModel):
    link: str

# Pydantic model for the response
class VideoLinkResponse(BaseModel):
    is_duplicate: bool
    duplicate_for: str = None

@app.post("/check-video-duplicate", response_model=VideoLinkResponse)
def check_video_duplicate_route(video: VideoLinkRequest):
    video_link = video.link

    # Download the video and save it as video.mp4
    dst = "video.mp4"
    urllib.request.urlretrieve(video_link, dst)

    # Check for duplicates
    is_duplicate, duplicate_uuid = check_video_duplicate(dst)

    # Remove the downloaded video file
    #os.remove(dst)

    # Formulate the response
    response = VideoLinkResponse(
        is_duplicate=is_duplicate,
        duplicate_for=duplicate_uuid if is_duplicate else ""
    )

    return response

def load_swagger_yaml(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

@app.get("")
async def read_root():
    return {"message": "Welcome to Video Duplicate Checker API. Go to /docs for Swagger documentation."}


@app.on_event("startup")
async def startup_event():
    # Load the Swagger YAML file
    openapi_schema = load_swagger_yaml("swagger.yaml")  # Adjust path as needed
    app.openapi_schema = openapi_schema


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
