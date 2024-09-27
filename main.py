from fastapi import FastAPI, HTTPException
from schema import VideoLinkRequest, VideoLinkResponse, VideoInsertResponse
import yaml
import urllib.request
from duplicates_db import check_video_duplicate
import uvicorn


app = FastAPI()


@app.post("/check-video-duplicate", response_model=VideoLinkResponse)
def check_video_duplicate_route(video: VideoLinkRequest):
    video_link = video.link
    uuid = video_link.split("/")[-1]
    print(uuid)

    # Download the video and save it as video.mp4
    dst = "video.mp4"
    urllib.request.urlretrieve(video_link, dst)

    # Check for duplicates
    is_duplicate, duplicate_uuid = check_video_duplicate(dst)

    # Remove the downloaded video file
    # os.remove(dst)

    # Formulate the response
    response = VideoLinkResponse(
        is_duplicate=is_duplicate,
        duplicate_for=duplicate_uuid if is_duplicate else ""
    )

    return response


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to Video Duplicate Checker API. Go to /docs for Swagger documentation."
    }


def load_swagger_yaml(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@app.on_event("startup")
async def startup_event():
    # Load the Swagger YAML file
    openapi_schema = load_swagger_yaml("swagger.yaml")  
    app.openapi_schema = openapi_schema


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
