from visualization import *


class Output:
    def __init__(self):
        self.visualizer = Visualization()
        pass

    def extract_video_subset(self, video_path, obj_data, output_path="outputs.mp4"):
        """Extracts a subset of a video based on frame indices and draws bounding boxes.

        Args:
            video_path (str): Path to the input video file.
            obj_data (list): A list containing bounding box data in the format [frame_ids:[], bboxes: []]
            output_path (str): Path to the outputs video file.
        """
        cap = cv2.VideoCapture(video_path)
        # Get video properties for outputs video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec (choose a suitable codec for your needs)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        obj_bboxes = obj_data[1]
        obj_frame_ids = obj_data[0]

        start_frame = obj_frame_ids[0]
        end_frame = obj_frame_ids[-1]

        current_frame = start_frame
        while current_frame <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()

            if ret:
                # Draw bounding boxes for each object in this frame
                for frame_id, bbox in zip(obj_frame_ids, obj_bboxes):
                    if frame_id == current_frame:
                        self.visualizer.draw_bounding_box(frame, bbox)

                out.write(frame)  # Write the frame to the outputs video
                current_frame += 1
            else:
                break

        cap.release()
        out.release()
