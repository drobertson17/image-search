import numpy as np
import webcolors
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.ImageFile import ImageFile
from sklearn.cluster import KMeans
from io import BytesIO
import base64
from typing import Any


def rgb_to_color_name(color):
    # Get a list of all webcolor names
    color_names = webcolors.names()
    
    colors = []
    for name in color_names:
        colors.append(np.array(webcolors.name_to_rgb(name)))
    colors = np.array(colors)

    # Compute closest
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))

    return color_names[index_of_smallest[0][0]]


class ImageProcessor:
    def __init__(self, image_path: str, vlm_pixel_limit: int = 999):
        self.path = image_path
        self.image: ImageFile = Image.open(image_path)
        self.vlm_pixel_limit = vlm_pixel_limit

    @property
    def vlm_image(self):
        buffered = BytesIO()
        width, height = self.image.size    # in pixels
        image = self.image.copy()
        if max(width, height) > self.vlm_pixel_limit:
            shrink_factor = self.vlm_pixel_limit/max(width, height)
            image = image.resize((
                int(image.width * shrink_factor),
                int(image.height * shrink_factor)
            ))
        
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_predominant_color(self, n_colors=1, return_name: bool = True):
        image = self.image.resize((self.image.width // 10, self.image.height // 10))
        image = image.convert('RGB')
        image_array = np.array(image)

        pixels = image_array.reshape(-1, 3)
        
        # Perform KMeans clustering to find the most common colors
        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(pixels)

        predominant_color = kmeans.cluster_centers_[0]

        if return_name:
            return rgb_to_color_name(tuple(map(int, predominant_color)))
        
        return tuple(map(int, predominant_color))

    def _json_serializable_type(self, value: Any):
        if isinstance(value, (str, int, float)):
            return value
        
        if isinstance(value, tuple):
            return [float(x) for x in value]
        
        try:
            return float(value)
        except:
            return str(value)


    @property
    def exif_data(self) -> dict:
        exif_data = {}
        try:
            image_exif =  self.image.getexif()
            for key, val in image_exif.items():
                val = self._json_serializable_type(val)
                if key in TAGS:
                    exif_data[TAGS[key]]=val
                else:
                    exif_data[key]=val
        except Exception as e:
            print(f"Error: {e}")

        return exif_data
