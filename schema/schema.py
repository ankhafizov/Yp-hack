from pydantic import BaseModel
import numpy as np


# Pydantic model for the request
class VideoLinkRequest(BaseModel):
    link: str


# Pydantic model for the response
class VideoLinkResponse(BaseModel):
    is_duplicate: bool
    duplicate_for: str = None


class VideoInsertResponse(BaseModel):
    embedding: list
