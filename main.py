from fastapi import FastAPI, HTTPException
from schema import VideoLinkRequest, VideoLinkResponse, VideoInsertResponse
import yaml
import os
import urllib.request
import uvicorn

from services import check_video_duplicate, insert_new_video
from elements.VideoElement import VideoElement

app = FastAPI()

with open("configs/app_config.yaml", "r") as file:
    config = yaml.safe_load(file)


def get_video(video_link):
    ### Получение пути до сохранения видео + UUID

    # Извлекаем UUID из ссылки
    uuid_with_extension = video_link.split("/")[-1]
    uuid = uuid_with_extension.split(".")[0]

    # Путь для сохранения видео
    video_folder = config["general"]["video_folder"]
    video_pth = os.path.join(video_folder, f"{uuid}.mp4")

    # Проверяем, существует ли папка, и создаем ее, если нет
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    return video_pth, uuid


@app.post("/check-video-duplicate", response_model=VideoLinkResponse)
async def check_video_duplicate_route(video: VideoLinkRequest):
    ### Эндпоинт по проверке дубликатов

    video_pth, uuid = get_video(video.link)

    # Скачиваем видео и сохраняем его
    try:
        urllib.request.urlretrieve(video.link, video_pth)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=400, detail="Ошибка 400: Bad Request (url не считавается)")

    # Проверяем на дубликат
    is_duplicate, duplicate_uuid = check_video_duplicate(
        VideoElement(video_path=video_pth, uuid=uuid)
    )

    # Удаляем загруженное видео
    if config["pipeline"]["delete_file_mp4"]:
        os.remove(video_pth)

    # Формируем Response
    response = VideoLinkResponse(
        is_duplicate=is_duplicate, duplicate_for=duplicate_uuid if is_duplicate else ""
    )

    return response


@app.post("/insert-video-db", response_model=VideoInsertResponse)
async def insert_new_video_route(video: VideoLinkRequest):
    ### Эндпоинт по проверке дубликатов

    video_pth, uuid = get_video(video.link)

    # Скачиваем видео и сохраняем его
    try:
        urllib.request.urlretrieve(video.link, video_pth)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=400, detail="Ошибка 400: Bad Request (url не считавается)")

    # Получаем эмбединг
    embedding = insert_new_video(VideoElement(video_path=video_pth, uuid=uuid))

    # Удаляем загруженное видео
    if config["pipeline"]["delete_file_mp4"]:
        os.remove(video_pth)

    # Формируем Response
    response = VideoInsertResponse(embedding=embedding)

    return response


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to Video Duplicate Checker API. Go to /docs for Swagger documentation."
    }


@app.on_event("startup")
async def startup_event():
    # Загрузчик swagger.yaml

    swagger_pth = config["general"]["swagger_pth"]
    with open(swagger_pth, "r", encoding="utf-8") as file:
        openapi_schema = yaml.safe_load(file)

    app.openapi_schema = openapi_schema


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=config["general"]["fastapi_port"], reload=True)
