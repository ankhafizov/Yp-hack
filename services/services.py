from nodes.VideoEmbeddingNode import VideoEmbeddingNode
from nodes.VectorDBNode import VectorDBNode
from nodes.VideoClassifierNode import VideoClassifierNode
from elements.VideoElement import VideoElement
import yaml


with open("configs/app_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Инициализация нод:
video_embedding_node = VideoEmbeddingNode(config)
vector_db_node = VectorDBNode(config)
video_classifier_node = VideoClassifierNode(config)


def check_video_duplicate(video_element: VideoElement):
    ### Функция реализует обработку видео и говорит есть ли дубликат (если есть то кто)

    video_element = video_embedding_node.process(video_element)
    video_element = vector_db_node.process_search(video_element)
    video_element = video_classifier_node.process(video_element)

    return video_element.is_dublicate, video_element.top_1_neighbour_uuid


def insert_new_video(video_element: VideoElement):
    ### Функция реализуект обработку видео и сохраняет эибеддинг в бд

    video_element = video_embedding_node.process(video_element)
    video_element = vector_db_node.process_insert(video_element)

    return list(video_element.embedding)
