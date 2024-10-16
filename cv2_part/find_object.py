# import os
# from pathlib import Path

# from PyQt5.QtCore import QLibraryInfo
# import cv2

# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
#     QLibraryInfo.PluginsPath
# )
# For except error qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/home/user/Documents/venv/lib/python3.11/site-packages/cv2/qt/plugins"
import os
from typing import TypeVar, NoReturn, Union, Sequence

import cv2

from .base_functools import *
from .downloload_part import loader

MatLike = TypeVar('MatLike')

def find_contours_of_cards(img, print_status: bool=False, thrash: int=200) -> Sequence[MatLike]:
    '''B/w image to input.'''
    if photo_info(img)['channel_count'] != 1:
        raise ValueError('No b/w image found.')
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    T, thresh_img = cv2.threshold(blurred, thrash, 255, cv2.THRESH_BINARY) 
    # Преобразуем в бинарное изображение. 
    # Первый аргумент -- преобразуемое изображение, второй аргумент -- пороговое значение, начиная от которого все последующие преобразуются до третьего аргумента.
    # cv2.THRESH_BINARY для того, чтобы второй параметр преобразовался ко третьему.
    if print_status:
        cv2.imshow('trash_binary', thresh_img)
        cv2.waitKey(0)
    (_, cnts) = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # cv2.RETR_EXTERNAL -- только внешние контуры. Для извлечения всех конткров применяется cv2.RETR_LIST.
    # Последний метод метод аппроксимации контура. cv2.CHAIN_APPROX_SIMPLE -- удаление лишних точек в целях экономиии памяти.
    return cnts

def find_coordinates_of_cards(img: MatLike, cnts: Sequence[MatLike]):
    cards_coordinates = {}
    for i in range(0, len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i]) # Ограничительные рамки каждого контура. 
        print()
        if w > 20 and h > 30: # Откидываем слишком маленькие изображения по условию.
            img_crop = img[y - 15:y + h + 15,
                             x - 15:x + w + 15]
            cards_name = find_features(img_crop)
            cards_coordinates[cards_name] = (x - 15, 
                     y - 15, x + w + 15, y + h + 15)
    return cards_coordinates

def find_features(img: MatLike) -> list:
    correct_matches_dct = {}
    directory = './dataSet/'
    for image in os.listdir(directory):
        img_for_comp = cv2.imread(directory+image, 0)
        orb = cv2.ORB.create()
        kp1, des1 = orb.detectAndCompute(img, None)
        kp2, des2 = orb.detectAndCompute(img_for_comp, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        correct_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                correct_matches.append([m])
                correct_matches_dct[image.split('.')[0]] = len(correct_matches)
        correct_matches_dct = dict(sorted(correct_matches_dct.items(), key=lambda item: item[1], reverse=True))
    return list(correct_matches_dct.keys())[0]

def draw_rectangle_aroud_cards(cards_coordinates, img):
    for key, value in cards_coordinates.items():
        rec = cv2.rectangle(img, (value[0], value[1]), 
                            (value[2], value[3]), 
                            (255, 255, 0), 2)
        cv2.putText(rec, key, (value[0], value[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (36, 255, 12), 1)
    cv2.imshow('Image', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    img = loader('./data_set/tokey_1.jpg', method_to_open=cv2.IMREAD_GRAYSCALE)
    print(find_contours_of_cards(img, print_status=True))
