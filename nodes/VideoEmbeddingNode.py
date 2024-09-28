from elements.VideoElement import VideoElement
import torch


from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
) 


alpha = 4

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

class VideoEmbeddingNode:
    def __init__(self, config: dict) -> None:
        self.embeding_method = config["embeding_method"]
        if self.embeding_method == "resnet_fifty":
           self.embeding_class = ResnetFiftyEmbeding(config[self.embeding_method])
        if self.embeding_method == "mvit":
             self.embeding_class = MvitEmbeding(config[self.embeding_method])
    def process(self, video_element: VideoElement):
        video_element.embedding = self.embeding_class.process(video_element.video_path)
        return video_element 

class ResnetFiftyEmbeding:
    def __init__(self, config: dict) -> None:
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        side_size = config["side_size"]
        crop_size = config["crop_size"]
        num_frames = config["num_frames"] 
        sampling_rate = config["sampling_rate"]
        frames_per_second = config["frames_per_second"]
        model_name = config["model_name"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=True)
        self.model.to(self.device)
        self.transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size),
                    PackPathway()
                ]
            ),
        )
        self.clip_duration = (num_frames * sampling_rate)/frames_per_second

    def process(self, video_path):
        start_sec = 0
        end_sec = start_sec + self.clip_duration 
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = self.transform(video_data)

        inputs = video_data["video"]
        inputs = [i.to(self.device)[None, ...] for i in inputs]

        return self.model(inputs).cpu().detach().numpy()[0].tolist()

class MvitEmbeding:
    def __init__(self, config: dict) -> None:
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        side_size = config["side_size"]
        crop_size = config["crop_size"]
        num_frames = config["num_frames"] 
        sampling_rate = config["sampling_rate"]
        frames_per_second = config["frames_per_second"]
        model_name = config["model_name"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=True)
        self.model.to(self.device)
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size)
                ]
            ),
        )
        self.clip_duration = (num_frames * sampling_rate)/frames_per_second

    def process(self, video_path):
        start_sec = 0
        end_sec = start_sec + self.clip_duration
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = self.transform(video_data)

        inputs = video_data["video"]
        inputs = video_data["video"].reshape(
            1,
            video_data["video"].shape[0],
            video_data["video"].shape[1],
            video_data["video"].shape[2],
            video_data["video"].shape[3],
        )
        inputs = inputs.to("cuda")

        return self.model(inputs).cpu().detach().numpy()[0].tolist()
