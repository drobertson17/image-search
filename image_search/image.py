import numpy as np
import webcolors
from PIL import Image
from sklearn.cluster import KMeans


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


def get_predominant_color(image_path, n_colors=1, return_name: bool = True):
    img = Image.open(image_path)
    img = img.resize((img.width // 10, img.height // 10))
    img = img.convert('RGB')
    img_array = np.array(img)

    pixels = img_array.reshape(-1, 3)
    
    # Perform KMeans clustering to find the most common colors
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    predominant_color = kmeans.cluster_centers_[0]

    if return_name:
        return rgb_to_color_name(tuple(map(int, predominant_color)))
    
    return tuple(map(int, predominant_color))