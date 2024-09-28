from elements.VideoElement import VideoElement
from pymilvus import MilvusClient
import json


class VectorDBNode:

    def __init__(self, config: dict) -> None:
        self.collection_name = config["collection_name"]
        self.dimension = config["dimension"]
        self.cosine_distance_treshold = config["cosine_distance_treshold"]
        self.metric_type = config["metric_type"]
        self.drop_db = config["drop_db"]
        self.host = config["host"]
        self.port = config["port"]

        # создание клиента и коллекции
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")

        if not self.client.has_collection(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type=self.metric_type,
            )
        # удаление бд по флагу
        if self.drop_db:
            _ = self.client.delete(collection_name= self.collection_name)
        self.num_video_element = 1

    def process_search(self, video_element: VideoElement):
        # метод опредляет является ли видео дубликатом по пороговому значению косинусного расстояния между векторами признаков видеозаписей
        res = self.client.search(
            collection_name=self.collection_name,
            data=[video_element.embedding],
            limit=1,
            search_params={"metric_type": self.metric_type, "params": {}},
            output_fields=["uid"],
        )
        if res and res[0]:
            res = res[0][0]
            video_element.top_1_neighbour_uuid = res["entity"]["uid"]
            metric_distance = float(res["distance"])
            video_element.is_dublicate = metric_distance >= self.cosine_distance_treshold
        return video_element

    def process_insert(self, video_element: VideoElement):
        # добавление новых объектов в бд
        data = [
            {
                "id": self.num_video_element,
                "vector": video_element.embedding,
                "created": video_element.timestamp_start_processing,
                "uid": video_element.uuid,
            }
        ]
        self.client.insert(collection_name=self.collection_name, data=data)
        self.num_video_element += 1
        return video_element
