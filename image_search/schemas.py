from typing import Protocol

from image_search.prompts import RequestData


class Model(Protocol):
    def __init__(self):
        ...

    def run(self, request: RequestData) -> str:
        ...