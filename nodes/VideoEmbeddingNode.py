from elements.VideoElement import VideoElement


class VideoEmbeddingNode:

    def __init__(self, config: dict) -> None:
        pass

    def process(self, video_element: VideoElement):
        video_element.embedding = [1, 1, 1, 1, 1]
        return video_element
