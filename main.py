import cv2
import obj_detection
import feature_extract
import feature_comparison
from database import DatabaseManager
from objs import Objects
from helpers import *
from visualization import *
import datetime
from features import Features
from output import Output
import tkinter as tk
from tkinter import filedialog

extracted_objects = []
MIN_DETECTION_CONFIDENCE = 0.7
extraction_interval = 5
is_performance_mode = True

"""
8. store obj_id, frame_id and bbox in a separate database
"""

# todo 1. streamlit progress bar if visualization is off
# todo 2. handle cases where an object is lost and then reappears in the scene.
# todo 5. preemptively save frames when similarity score is very high (95%+).
#  else save the top k results only
# todo 6. load a quantized resnet if performance mode is off


"""
Database looks like this
objects: obj_id (INT), cls (INT), conf (REAL), bbox (TEXT),  
            frame_ids = [frame_id], similarity_score (REAL)> obj_id 0 for target image
features: obj_id (INT), features (BLOB)
bboxes: obj_id INTEGER, frame_id INTEGER, bbox TEXT, PRIMARY KEY (obj_id, frame_id),
"""

# save some essential metadata
metadata = {}

# basic local helpers


def update_metadata(key, value):
    metadata[key] = value


def iterate_detection_result(res, tracking_on):
    bbox = res[:4].to(int).tolist()
    conf = round(float(res[-2]), 2)
    cls = int(res[-1])
    if tracking_on:
        obj_id = int(res[4])
        return obj_id, cls, conf, bbox
    return cls, conf, bbox


def should_extract_features(obj):
    """determine whether to extract features for a given object."""
    if is_performance_mode:
        return (obj.frame_ids[-1] - obj.frame_ids[0]) % extraction_interval == 0
    else:
        return obj.obj_id not in extracted_objects


def get_time():
    cur_time = datetime.datetime.now()
    strf = "%-d-%-m-%H:%M"
    time_str = datetime.datetime.strftime(cur_time, strf)
    return time_str


def filter_by_color(objects, target_color):
    """Filters a dictionary of objects by a given color."""
    filtered_objects = {}
    for obj_id, obj in objects.items():
        if is_similar_color(obj.color, target_color):
            filtered_objects[obj_id] = obj
    return filtered_objects


def get_high_rank(obj_dict, color):
    filtered_obj_dict = filter_by_color(obj_dict, color)
    del obj_dict
    sorted_dict_id = sorted(filtered_obj_dict, key=lambda x: filtered_obj_dict[x].similarity, reverse=True)
    # sorted_dict_id = sorted(obj_dict, key=lambda x: obj_dict[x].similarity, reverse=False)
    return sorted_dict_id


def get_input(title):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_name = filedialog.askopenfilename(title=title)
    return file_name


# basic local helpers end

""" Input logic here """
input_img_path = get_input("Select an image")
input_video_path = get_input("Select a video")


update_metadata("input_video_path", input_video_path)
update_metadata("input_img_path", input_img_path)

""" Input logic end """


def main():
    database_name = get_time()
    db = DatabaseManager(database_name)
    # create objects and features tables here
    db.create_objects_table()
    db.create_features_table()
    # todo create bboxes db

    # initialize detector, extractor, feature comparison, visualizer objects
    detector = obj_detection.ObjDetection()
    feature_extractor = feature_extract.FeatureExtract("inception")
    f_comparison = feature_comparison.FeatureComparison(threshold=0.8)  # I know f_comparison is a shitty name
    visualizer = Visualization()

    # logic for processing the target image
    # todo move it elsewhere, or make it prettier idk
    im_result = detector.detect(input_img_path, isImage=True)
    # save cropped object image without background
    best_result = save_target_obj(im_result)

    # assign object class, confidence score to variables
    cls, conf, coord = iterate_detection_result(best_result, tracking_on=False)
    target_obj_id = 0
    target_frame_index = 0

    # extract target object's features
    target_arr = im_result[0].orig_img
    target_obj = crop_object(coord, target_arr)
    target_color = get_dominant_color(target_obj, 3)

    target_feature = feature_extractor.extract_vector(target_obj)
    squeezed_target = reshape_target(target_feature)
    # squeezed_target = reshape_feature(target_feature)

    update_metadata("target_obj_id", target_obj_id)
    update_metadata("target_feature_shape", squeezed_target.shape)

    # initialize objects dict with object instances inside
    objects = {target_obj_id: Objects(obj_id=target_obj_id,
                                      cls=cls,
                                      conf=conf,
                                      color=target_color,
                                      similarity=1)}
    objects[target_obj_id].add_bbox(coord)
    objects[target_obj_id].add_frame_index(target_frame_index)

    # initialize object features dict
    # this dict will contain *best* values
    obj_features = {target_obj_id: Features(target_obj_id)}
    obj_features[target_obj_id].set_feature(feature=squeezed_target)
    # logic for processing the target image END

    # Loop through the video frames
    cap = cv2.VideoCapture(input_video_path)

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, with tracking on (persist=true)
            results = detector.detect(frame, classes=[cls], tracker="bytetrack.yaml", persist=True)

            current_frame = results[0].orig_img
            current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # frame index starts with 1

            # iterate over one frame's results start
            detection_data = results[0].boxes.data
            for result in detection_data:
                obj_id, cls, conf, bbox = iterate_detection_result(result, tracking_on=True)

                if obj_id in objects.keys():
                    current_obj = objects[obj_id]
                    current_obj.add_frame_index(current_frame_index)
                    current_obj.add_bbox(bbox)
                else:
                    dominant_color = get_dominant_color(crop_object(bbox, current_frame), 3)
                    current_obj = Objects(obj_id=obj_id, cls=cls, conf=conf, color=dominant_color)
                    current_obj.add_frame_index(current_frame_index)
                    current_obj.add_bbox(bbox)
                    objects[obj_id] = current_obj

                    # get dominant color of the object

                # extract feature of the current object
                if should_extract_features(current_obj):
                    cropped_object = crop_object(bbox, current_frame)
                    feature = feature_extractor.extract_vector(cropped_object)
                    # squeeze to remove dimension of 1
                    squeezed_feature = reshape_feature(feature)
                    similarity = f_comparison.cosine_similarity_matrix(squeezed_target, squeezed_feature)[0]
                    # similarity = f_comparison.euclidian_distance(squeezed_target, squeezed_feature)[0]
                    current_obj.similarity = similarity
                    # why did I do this????
                    if obj_id not in obj_features:
                        obj_features[obj_id] = Features(obj_id)
                        obj_features[obj_id].set_feature(feature)
                    elif similarity > objects[obj_id].similarity:
                        obj_features[obj_id].set_feature(feature)
                        objects[obj_id].similarity = similarity

                    if "features_shape" not in metadata:
                        features_shape = feature.shape
                        update_metadata("features_shape", features_shape)

                if visualizer.is_on:
                    visualizer.visualize_object(frame=frame, obj=current_obj)

            # iterate over one frame's results end
            if visualizer.is_on:
                visualizer.display_frame(frame=frame)

            # TODO Visualize the results on the frame
            #  have a toggle for visualization
            #  show bbox, similarity, id

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    # insert objects, features into database
    for obj_id in objects.keys():
        obj = objects[obj_id]
        feats = obj_features[obj_id].feature
        sim = round(float(obj.similarity), 2)
        print(obj.color)
        db.insert_to_objects(obj_id=obj.obj_id,
                             cls=obj.cls,
                             conf=obj.conf,
                             bbox=obj.bbox,
                             frame_ids=obj.frame_ids,
                             similarity=sim)
        db.insert_to_features(obj_id=obj_id, features=feats)

    write_metadata_to_file(metadata)
    print("Insertion into objects table successful")

    print("saving output...")
    video_output = Output()
    print(target_color)
    similar_obj_ids = get_high_rank(objects, target_color)
    top_k = 3
    for similar_obj_id in similar_obj_ids[:top_k + 1]:
        if similar_obj_id == 0:
            continue
        bboxes, frames = objects[similar_obj_id].bbox, objects[similar_obj_id].frame_ids
        video_output.extract_video_subset(metadata["input_video_path"],
                                          [frames, bboxes], output_path=f"outputs/output_{similar_obj_id}.mp4")


main()
