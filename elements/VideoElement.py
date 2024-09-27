import numpy as np
import time

class VideoElement:
    # Класс, содержаций информацию о видео
    def __init__(
        self,
        video_path: str,
        uuid: str,
        embedding: np.ndarray | None = None,
        top_1_neighbour_uuid: str | None = None,
        is_dublicate: bool = False
    ) -> None:
        self.video_path = video_path  # Путь к видео
        self.uuid = uuid  # UUID данного видео
        self.embedding = embedding  # Значение эмбеддинга
        self.top_1_neighbour_uuid = top_1_neighbour_uuid  # UUID ближайшего соседа
        self.is_dublicate = is_dublicate  # Является ли видео дубликатом
        self.timestamp_start_processing = time.time()  # Время в момент начала обработки видео unix формат (в секундах)