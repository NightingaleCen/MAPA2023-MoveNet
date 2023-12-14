import cv2
import time
from movenet import MoveNet, KEYPOINT_DICT, KEYPOINT_EDGE_INDS_TO_COLOR
import tensorflow as tf
from cropping import *
from pose_classifier import PoseClassifier, num2label


class PoseEstimationCamera:
    def __init__(
        self,
        width,
        height,
        model_name="movenet_thunder",
        classifier_ckpt=None,
        default_camera=1,
        keypoint_threshold=0.11,
    ) -> None:
        self.width = width
        self.height = height
        self.default_camera = default_camera
        self.keypoint_threshold = keypoint_threshold

        self._get_available_cameras()
        self.net = MoveNet(model_name)
        if classifier_ckpt != None:
            self.classifier = PoseClassifier.load_model(
                file_path=classifier_ckpt, input_shape=(17, 3)
            )
        self.enter()

    def _get_available_cameras(self):
        self.available_cameras = []
        index = 0

        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            self.available_cameras.append(index)
            cap.release()
            index += 1

        self.camera_num = len(self.available_cameras)

    def enter(self):
        self.cap = cv2.VideoCapture(self.default_camera)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        current_camera = self.default_camera

        crop_region = init_crop_region(self.height, self.width)

        display_keypoints = False
        enable_pose_classification = False

        while True:
            start_time = time.time()

            ret, frame = self.cap.read()
            H, W, C = frame.shape

            # inferring with the cropping algorithm
            input_tensor = self.frame_preprocess(frame)
            keypoints_with_scores = run_inference(
                self.net,
                input_tensor,
                crop_region,
                crop_size=[self.net.input_size, self.net.input_size],
            )
            self.draw_points_and_lines(frame, keypoints_with_scores)
            crop_region = determine_crop_region(keypoints_with_scores, H, W)

            if display_keypoints:  # display the keypoint information
                keypoint_text_y = 60
                for point in KEYPOINT_DICT.keys():
                    if (
                        keypoints_with_scores[0][0][KEYPOINT_DICT[point]][2]
                        < self.keypoint_threshold
                    ):
                        coordinate_string = "NaN"
                    else:
                        point_coord_x, point_coord_y = (
                            keypoints_with_scores[0][0][KEYPOINT_DICT[point]][1],
                            keypoints_with_scores[0][0][KEYPOINT_DICT[point]][0],
                        )
                        coordinate_string = "({0:.2f},{1:.2f})".format(
                            point_coord_x, point_coord_y
                        )
                    cv2.putText(
                        frame,
                        text=f"{point}: {coordinate_string}",
                        org=(10, keypoint_text_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                    keypoint_text_y += int(1.5 * 0.5 * 20)

            # enable pose classification
            if enable_pose_classification:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_color = (0, 0, 255)
                thickness = 2

                all_points_visible = True
                for point in [
                    "left_shoulder",
                    "right_shoulder",
                    "left_hip",
                    "right_hip",
                ]:
                    if (
                        keypoints_with_scores[0][0][KEYPOINT_DICT[point]][2]
                        < self.keypoint_threshold
                    ):
                        all_points_visible = False
                        break

                # classify only when all points are visible
                if all_points_visible:
                    reshaped_keypoints_with_scores = keypoints_with_scores.reshape(
                        -1, 17, 3
                    )
                    pred = self.classifier.predict(reshaped_keypoints_with_scores)[0]
                    pred_class = np.argmax(pred)

                    label = f"{num2label[pred_class]}"
                    confidence = f"{pred[pred_class]}"

                    text_size1 = cv2.getTextSize(label, font, font_scale, thickness)[0]
                    text_size2 = cv2.getTextSize(
                        confidence, font, font_scale, thickness
                    )[0]

                    text_x1 = (W - text_size1[0]) // 2
                    text_y1 = text_size1[1]
                    text_x2 = (W - text_size2[0]) // 2
                    text_y2 = text_y1 + text_size1[1] + int(0.5 * text_size2[1])

                    cv2.putText(
                        frame,
                        label,
                        (text_x1, text_y1),
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        confidence,
                        (text_x2, text_y2),
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        cv2.LINE_AA,
                    )
                else:
                    text = "Show your core body"
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                    text_x1 = (W - text_size[0]) // 2
                    text_y1 = text_size[1]
                    cv2.putText(
                        frame,
                        text,
                        (text_x1, text_y1),
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        cv2.LINE_AA,
                    )
            # show fps
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(
                frame,
                text=f"FPS: {int(fps)}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # show threshold
            info_text = "Point Threshold: {:.2f}".format(self.keypoint_threshold)
            text_width, _ = cv2.getTextSize(
                info_text,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=2,
            )[0]
            cv2.putText(
                frame,
                text=info_text,
                org=(W - 10 - text_width, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Press q to quit
                break
            elif key == ord("c"):  # Press c to switch camera (if any)
                self.cap.release()
                current_camera = (current_camera + 1) % self.camera_num
                self.cap = cv2.VideoCapture(current_camera)
                self.cap.set(3, self.width)
                self.cap.set(4, self.height)
            elif key == ord("p"):  # Press c to switch on pose classifier
                enable_pose_classification = not enable_pose_classification

            elif key == ord("o"):  # Press o to display the keypoint information
                display_keypoints = not display_keypoints
            elif (
                key == ord("]") and self.keypoint_threshold < 1
            ):  # Press ] to raise keypoint threshold
                self.keypoint_threshold += 0.01
            elif (
                key == ord("[") and self.keypoint_threshold > 0
            ):  # Press [ to lower keypoint threshold
                self.keypoint_threshold -= 0.01

        self.cap.release()
        cv2.destroyAllWindows()

    def frame_preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(frame)
        return input_tensor

    def draw_points_and_lines(self, frame, keypoints_with_scores):
        H, W, C = frame.shape

        for line in KEYPOINT_EDGE_INDS_TO_COLOR.keys():
            p1, p2 = line
            if (
                keypoints_with_scores[0][0][p1][2] >= self.keypoint_threshold
                and keypoints_with_scores[0][0][p2][2] >= self.keypoint_threshold
            ):
                p1_coord = (
                    int(keypoints_with_scores[0][0][p1][1] * W),
                    int(keypoints_with_scores[0][0][p1][0] * H),
                )
                p2_coord = (
                    int(keypoints_with_scores[0][0][p2][1] * W),
                    int(keypoints_with_scores[0][0][p2][0] * H),
                )
                cv2.line(frame, p1_coord, p2_coord, color=(0, 255, 0), thickness=2)

        for point in KEYPOINT_DICT.keys():
            if (
                keypoints_with_scores[0][0][KEYPOINT_DICT[point]][2]
                >= self.keypoint_threshold
            ):
                coord_x, coord_y = (
                    int(keypoints_with_scores[0][0][KEYPOINT_DICT[point]][1] * W),
                    int(keypoints_with_scores[0][0][KEYPOINT_DICT[point]][0] * H),
                )
                cv2.circle(
                    frame,
                    center=(coord_x, coord_y),
                    radius=5,
                    color=(0, 255, 255),
                    thickness=-1,
                )


if __name__ == "__main__":
    PoseEstimationCamera(
        width=1920,
        height=1080,
        model_name="movenet_thunder",
        classifier_ckpt="pose_classifier.h5",
        default_camera=1,
        keypoint_threshold=0.3,
    )
