from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import cv2
import numpy as np
import dataclasses
import math
from typing import Tuple, Union

VIDEO_INPUT = 1
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)


@dataclasses.dataclass
class DrawingSpec:
    color: Tuple[int, int, int] = WHITE_COLOR

    thickness: int = 2
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


##


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def ty(image, list1):
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        ####
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ###
                list2 = []
                if not face_landmarks:
                    return
                if image.shape[2] != _BGR_CHANNELS:
                    raise ValueError("Input image must contain three channel bgr data.")
                image_rows, image_cols, _ = image.shape
                idx_to_coordinates = {}
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if (
                        landmark.HasField("visibility")
                        and landmark.visibility < _VISIBILITY_THRESHOLD
                    ) or (
                        landmark.HasField("presence")
                        and landmark.presence < _PRESENCE_THRESHOLD
                    ):
                        continue
                    landmark_px = _normalized_to_pixel_coordinates(
                        landmark.x, landmark.y, image_cols, image_rows
                    )
                    if landmark_px:
                        idx_to_coordinates[idx] = landmark_px
                try:
                    for a in list1:
                        list2.append(list(idx_to_coordinates[a]))
                except Exception:
                    continue
                return list2


##
cap = cv2.VideoCapture(VIDEO_INPUT)

if not cap.isOpened():
    print("Error: Unable to open video capture.")
    exit()
w = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) / 2)
h = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / 30)
h1 = int(h * 12)
h2 = int(h * 28)

print(h)
while True:
    ret, img = cap.read()
    frame = img.copy()
    for i in range(7):
        cv2.line(frame, (w, h + h * 4 * i), (w, 4 * h * (1 + i)), (0, 0, 255), 5)
    cv2.line(frame, (w - 112, h1), (w + 112, h1), (0, 0, 255), 3)
    cv2.line(frame, (w - 50, h2), (w + 50, h2), (0, 0, 255), 5)

    cv2.imshow("Framing", cv2.flip(frame, 1))

    if cv2.waitKey(1) & 0xFF == ord(" "):
        break


# cv2.imwrite("eye.jpg", img)


cap.release()


##

# img = cv2.imread("eye.jpg")
cap = cv2.VideoCapture(VIDEO_INPUT)

indexes_triangles = []

# Face 1
listR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
listL = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
for i in range(2):
    if i == 0:
        listF = listR
    elif i == 1:
        listF = listL
    landmarks3 = ty(img, listF)
    if landmarks3 is None:
        continue

    points = np.array(landmarks3, np.int32)

    # Delaunay triangulation
    rect = cv2.boundingRect(points)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks3)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

        if i == 0:
            indexes_trianglesR = indexes_triangles
            pointsR = points
            landmarks3B = landmarks3
        elif i == 1:
            indexes_trianglesL = indexes_triangles
            pointsL = points
            landmarks3L = landmarks3


while True:
    _, img2 = cap.read()
    img2_new_face = np.zeros_like(img2)

    for s in range(2):
        if s == 0:
            listF = listR
            indexes_triangles = indexes_trianglesR
            points = pointsR
            landmarks3 = landmarks3B
        elif s == 1:
            listF = listL
            indexes_triangles = indexes_trianglesL
            points = pointsL
            landmarks3 = landmarks3L

        landmarks_points2 = ty(img2, listF)
        if landmarks_points2:
            points2 = np.array(landmarks_points2, np.int32)
            for triangle_index in indexes_triangles:
                # Triangulation of the first face
                tr1_pt1 = landmarks3[triangle_index[0]]
                tr1_pt2 = landmarks3[triangle_index[1]]
                tr1_pt3 = landmarks3[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                rect1 = cv2.boundingRect(triangle1)
                (x, y, w, h) = rect1
                cropped_triangle = img[y : y + h, x : x + w]
                cropped_tr1_mask = np.zeros((h, w), np.uint8)

                points = np.array(
                    [
                        [tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y],
                    ],
                    np.int32,
                )

                cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
                img2 = img2.copy()


                cv2.fillPoly(img2, [np.array(landmarks_points2, np.int32)], (0, 0, 0))

                tr2_pt1 = landmarks_points2[triangle_index[0]]
                tr2_pt2 = landmarks_points2[triangle_index[1]]
                tr2_pt3 = landmarks_points2[triangle_index[2]]
                triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                rect2 = cv2.boundingRect(triangle2)
                (x, y, w, h) = rect2

                cropped_tr2_mask = np.zeros((h, w), np.uint8)

                points2 = np.array(
                    [
                        [tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y],
                    ],
                    np.int32,
                )

                cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

                points = np.float32(points)
                points2 = np.float32(points2)
                M = cv2.getAffineTransform(points, points2)
                warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                warped_triangle = cv2.bitwise_and(
                    warped_triangle, warped_triangle, mask=cropped_tr2_mask
                )

                img2_new_face_rect_area = img2_new_face[y : y + h, x : x + w]
                img2_new_face_rect_area_gray = cv2.cvtColor(
                    img2_new_face_rect_area, cv2.COLOR_BGR2GRAY
                )
                _, mask_triangles_designed = cv2.threshold(
                    img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV
                )
                warped_triangle = cv2.bitwise_and(
                    warped_triangle, warped_triangle, mask=mask_triangles_designed
                )
                img2_new_face_rect_area = cv2.add(
                    img2_new_face_rect_area, warped_triangle
                )

                img2_new_face[y : y + h, x : x + w] = img2_new_face_rect_area

    result = cv2.add(img2, img2_new_face)

    cv2.imshow("Video", cv2.flip(result, 1))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()
