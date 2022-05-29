import os
import cv2
import random
import numpy as np

from tqdm import tqdm
from typing import List
from typing import Tuple


PATHS = ["/path/to/dataset1",
         "/path/to/dataset2",
         "/path/to/dataset3"]


class Dataset:
    def __init__(self, path: str) -> None:
        """
        :param path: path to dataset.
        """
        filepaths = [os.path.join(path, filename) for filename in os.listdir(path)]
        names_path_list = [fp for fp in filepaths if fp.endswith(".names")]
        obj_path_list = [fp for fp in filepaths if os.path.isdir(fp) and "Visualisation" not in fp]

        assert len(names_path_list) == 1 and len(obj_path_list) == 1, \
            "The directory should contain one .names file and one folder with images and annotations!"

        self.path = path
        self.obj_path = obj_path_list[0]
        with open(names_path_list[0], "r") as rn:
            self.names = {id: name.replace("\n", "") for id, name in enumerate(rn.readlines())}
        self.colors = {0: (0, 0, 128),
                       1: (0, 128, 0),
                       2: (0, 128, 128),
                       3: (128, 0, 0),
                       4: (128, 0, 128),
                       5: (128, 128, 0),
                       6: (128, 128, 128),
                       7: (0, 0, 64)}

    def create_visualisation(self) -> None:
        """ Create visualisation of dataset. """
        vis_path = os.path.join(self.path, "Visualisation")
        if os.path.exists(vis_path):
            os.rmdir(vis_path)
        os.makedirs(vis_path)

        obj_files = os.listdir(self.obj_path)
        print("Visualisation: ")
        for filename in tqdm(obj_files):
            if not filename.endswith(".txt"):
                marked_image = self._mark_image(filename)
                save_path = os.path.join(vis_path, f"marked_{filename}")
                cv2.imwrite(save_path, marked_image)

    def _mark_image(self, filename: str) -> np.ndarray:
        """
        Create image with marks.

        :param filename: image filename.
        :return: image with marks.
        """
        image = cv2.imread(os.path.join(self.obj_path, filename))
        self.cur_image = image
        objects = self._get_objects(filename)

        h, w, d = image.shape
        # Image with translucent fills.
        added_img = np.zeros([h, w, d], dtype=np.uint8)
        screen_h = 1080
        screen_w = 1920
        # Thickness of lines.
        scale = np.min([float(screen_h) / float(h), float(screen_w) / float(w)])
        for obj in objects:
            image = self._draw_object(obj, image, added_img, scale)
        # Overlay polygon fills.
        cv2.addWeighted(added_img, 0.8, image, 0.9, 0, image)
        return image

    def _get_objects(self, img_filename: str) -> List:
        """
        Get data of objects from txt-file.

        :param img_filename: image filename.
        :return: list of objects (dictionaries).
        """
        img_ext = img_filename.split(".")[-1]
        txt_filename = img_filename.replace(img_ext, "txt")
        with open(os.path.join(self.obj_path, txt_filename), "r") as tr:
            lines = tr.readlines()

        objects = []
        for line in lines:
            name_id, x, y, w, h = [float(value) for value in line.replace("\n", "").split(" ")]
            object = {"name": self.names[int(name_id)],
                      "points": self._convert_coords([x, y, w, h])}
            objects.append(object)
        return objects

    def _convert_coords(self, coords: List) -> np.ndarray:
        """
        Convert coords from Yolo format:
            `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
            `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
        To:
            `([x1, y1], [x2, y2], [x3, y3], [x4, y4])`, e.g. ([97, 12], [247, 12], [247, 212], [97, 212]).

        :param coords: coords in Yolo format.
        :return: converted coords.
        """
        x, y, w, h = coords
        img_h, img_w = self.cur_image.shape[:2]
        object_width = w * img_w
        object_height = h * img_h
        x_sum = x * img_w * 2
        y_sum = y * img_h * 2
        x_min = round((x_sum - object_width) / 2)
        y_min = round((y_sum - object_height) / 2)
        x_max = round(x_sum - x_min)
        y_max = round(y_sum - y_min)
        [x1, y1], [x2, y2], [x3, y3], [x4, y4] = [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
        xy = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype(int)  # Polygon
        return xy

    def _draw_object(self, obj: dict, src_img: np.ndarray, added_img: np.ndarray, scale: np.float) -> np.ndarray:
        """
        Draw labeled object on image.
        :param obj: dictionary with data about object.
        :param src_img: image to draw edging.
        :param added_img: image to draw fills.
        :param scale: scale of thickness edging.
        :return: marked image.
        """
        class_name = obj['name']
        # Get key of value (`class_name`) of dictionary.
        color = self._get_color(list(self.names.keys())[list(self.names.values()).index(class_name)])
        coords = obj['points']
        # Draw the fill.
        cv2.fillConvexPoly(added_img, coords, color)
        # Draw the rectangle.
        cv2.polylines(src_img, [coords], True, color, thickness=int(4.0 / scale))
        # Draw the text.
        x1, y1 = coords[0]
        src_img = self._draw_text_box(src_img, class_name, x1, y1)
        return src_img

    def _get_color(self, id: int) -> Tuple[int, int, int]:
        """
        Return the color of object.
        :param id: id of object.
        :return: color.
        """
        if id in self.colors.keys():
            return self.colors[id]
        new_color = self.colors[0]
        colors_values = self.colors.values()
        while new_color in colors_values:
            new_color = (random.randint(20, 230), random.randint(20, 230), random.randint(20, 230))
        self.colors[len(self.colors)] = new_color
        return new_color

    @staticmethod
    def _draw_text_box(img: np.array, text: str, x: int, y: int,
                       font_color=(255, 255, 255), back_color=(0, 0, 0), font_scale=0.5, thickness=1) -> np.ndarray:
        """
        Draw small text box on image.
        :param img: image to draw.
        :param text: text to draw.
        :param x: coord x.
        :param y: coord y.
        :param font_color: color of font.
        :param back_color: color of background.
        :param font_scale: scale of font.
        :param thickness: thickness of text.
        :return:
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Get the width and height of the text box.
        t_w, t_h = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
        # Make the coords of the box with a small padding of two pixels.
        box_coords = [(int(x), int(y + 5)), (int(x + t_w), int(y - t_h))]
        cv2.rectangle(img, box_coords[0], box_coords[1], back_color, cv2.FILLED)
        cv2.putText(img, str(text), (int(x + 1), int(y + 1)),
                    font, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        cv2.putText(img, str(text), (int(x), int(y)),
                    font, fontScale=font_scale, color=font_color, thickness=thickness)
        return img


def main():
    for path in PATHS:
        ds = Dataset(path)
        print(f'Dataset {path.split("/")[-1]}')
        ds.create_visualisation()


if __name__ == "__main__":
    main()
