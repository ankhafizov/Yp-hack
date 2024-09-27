from fastapi import FastAPI, HTTPException
from schema import VideoLinkRequest, VideoLinkResponse, VideoInsertResponse
import yaml
import os
import urllib.request
import uvicorn

from nodes.VideoEmbeddingNode import VideoEmbeddingNode
from nodes.VectorDBNode import VectorDBNode
from nodes.VideoClassifierNode import VideoClassifierNode
from elements.VideoElement import VideoElement


app = FastAPI()

with open("configs/app_config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Инициализация нод:
video_embedding_node = VideoEmbeddingNode(config)
vector_db_node = VectorDBNode(config)
video_classifier_node = VideoClassifierNode(config)

def check_video_duplicate(video_element: VideoElement):
    ### Функция реализуект обработку видео и говорит есть ли дубликат (если есть то кто)

    video_element = video_embedding_node.process(video_element)
    video_element = vector_db_node.process_search(video_element)
    video_element = video_classifier_node.process(video_element)
    
    return video_element.is_dublicate, video_element.top_1_neighbour_uuid


@app.post("/check-video-duplicate", response_model=VideoLinkResponse)
def check_video_duplicate_route(video: VideoLinkRequest):
    ### Эндпоинт по проверке дубликатов 
    
    video_link = video.link

    # Извлекаем UUID из ссылки
    uuid_with_extension = video_link.split("/")[-1]
    uuid = uuid_with_extension.split(".")[0]

    # Путь для сохранения видео
    video_pth = os.path.join(config["general"]["video_folder"], f"{uuid}.mp4")

    # Проверяем, существует ли папка, и создаем ее, если нет
    if not os.path.exists(config["general"]["video_folder"]):
        os.makedirs(config["general"]["video_folder"])

    # Скачиваем видео и сохраняем его
    urllib.request.urlretrieve(video_link, video_pth)

    # Check for duplicates
    is_duplicate, duplicate_uuid = check_video_duplicate(
        VideoElement(video_path=video_pth, uuid=uuid)
    )

    # Remove the downloaded video file
    if config["pipeline"]["delete_file_mp4"]:
        os.remove(video_pth)

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
