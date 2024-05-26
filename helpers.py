from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import cv2
from sklearn.cluster import KMeans

color_map = {
    "black": (0, 0, 0),
    "silver": (192, 192, 192),
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
    "maroon": (128, 0, 0),
    "red": (255, 0, 0),
    "olive": (128, 128, 0),
    "yellow": (255, 255, 0),
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "teal": (0, 128, 128),
    "aqua": (0, 255, 255),
    "navy": (0, 0, 128),
    "blue": (0, 0, 255),
    "purple": (128, 0, 128),
    "fuchsia": (255, 0, 255),
    "brown": (165, 42, 42),
    "beige": (245, 245, 220),
    "coral": (255, 127, 80),
    "salmon": (250, 128, 114),
    "orange": (255, 165, 0),
    "gold": (255, 215, 0),
    "khaki": (240, 230, 140),
    "cyan": (0, 255, 255),
    "sky blue": (135, 206, 235),
    "turquoise": (64, 224, 208),
    "pink": (255, 192, 203),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "indigo": (75, 0, 130),
    "plum": (221, 160, 221),
    "rose": (255, 0, 127),
    "magenta": (255, 0, 255),
    "tan": (210, 180, 140),
    "light gray": (211, 211, 211),
    "dark gray": (169, 169, 169),
    "light blue": (173, 216, 230),
    "peach": (255, 218, 185),
    "mint": (189, 252, 201),
    "pale green": (152, 251, 152),
    "light yellow": (255, 255, 224),
    "lemon": (255, 250, 205),
    "olive drab": (107, 142, 35)
}


def preprocess_image(image):
    # image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    return pixels


def get_dominant_color(image, n_clusters=2):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = preprocess_image(image)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels)
    dominant_cluster = np.argmax(cluster_counts)
    dominant_color = kmeans.cluster_centers_[dominant_cluster]
    return tuple(dominant_color.astype(int))


# print(get_dominant_color(cv2.imread("files/damas.jpeg")))

def calculate_color_distance(color1, color2):
    return np.sqrt(
        np.sum(
            (np.array(color1) - np.array(color2)) ** 2
        )
    )


def is_similar_color(color1, color2, threshold=20):
    distance = calculate_color_distance(color1, color2)
    return distance <= threshold


def get_closest_color_name(rgb_value):
    min_distance = float("inf")
    closest_color_name = None
    print(rgb_value)
    for name, color in color_map.items():
        distance = calculate_color_distance(rgb_value, color)
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
    return closest_color_name


def convert_bbox(bboxes):
    output = []
    for bbox in bboxes.split("-"):
        arr_data = ast.literal_eval(bbox)
        output.append(arr_data)
    return output


def convert_list_to_string(arr):
    return "-".join([str(x) for x in arr])


def convert_string_to_list(arr_str):
    return [int(x) for x in arr_str.split("-")]


# TODO implement better logic for picking target object
def save_target_obj(results):
    #      x          y             x           y         conf      cls
    # [6.0955e+01, 2.4356e+02, 2.8908e+02, 3.8819e+02, 9.3720e-01, 2.0000e+00]
    max_conf = 0
    max_conf_res = []
    for result in results[0].boxes.data:
        if result[4] > max_conf:
            max_conf_res = result
            max_conf = result[4]

    image_result = max_conf_res
    im = results[0].orig_img[:, :, ::-1]
    img = Image.fromarray(im)
    x, y, m, n = image_result[:4].to(int).tolist()
    cropped_image = img.crop((x, y, m, n))
    cropped_image.save("outputs/target_obj.png")
    return image_result


# todo maybe remove
def from_bytes(bytes_data, shape):
    return np.frombuffer(bytes_data, dtype=np.float32).reshape(shape)

def crop_object(bbox, frame):
    """crop object from frame given bbox"""
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


# cls, feature_shape, frame_shape, img_path, input_video_path, num_of_objects_detected,
def write_metadata_to_file(metadata):
    """Metadata handling"""
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f)


def reshape_target(arr):
    return arr.squeeze()


def reshape_feature(arr):
    """remove unnecessary dimensions"""
    return arr.reshape(1, -1)
