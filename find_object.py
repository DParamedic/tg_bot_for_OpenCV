# import os
# from pathlib import Path

# from PyQt5.QtCore import QLibraryInfo
# import cv2

# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
#     QLibraryInfo.PluginsPath
# )
# For except error qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/home/user/Documents/venv/lib/python3.11/site-packages/cv2/qt/plugins"

import cv2
from typing import TypeVar, NoReturn, Union

from base_functools import *
from downloload_part import loader

MatLike = TypeVar('MatLike')

def find_contours_of_cards(img, print_status: bool=False, thrash: int=200) -> MatLike:
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
        return
    cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # cv2.RETR_EXTERNAL -- только внешние контуры. Для извлечения всех конткров применяется cv2.RETR_LIST.
    # Последний метод метод аппроксимации контура. cv2.CHAIN_APPROX_SIMPLE -- удаление лишних точек в целях экономиии памяти.
    return cnts

def find_coordinates_of_cards(img, cnts):
    cards_coordinates = {}
    for i in range(0, len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])
        if w > 20 and h > 30:
            img_crop = img[y - 15:y + h + 15,
                             x - 15:x + w + 15]
            cards_name = find_features(img_crop)
            cards_coordinates[cards_name] = (x - 15, 
                     y - 15, x + w + 15, y + h + 15)
    return cards_coordinates

if __name__ == "__main__":
    img = loader('./data_set/tokey_1.jpg', method_to_open=cv2.IMREAD_GRAYSCALE)
    print(find_contours_of_cards(img, print_status=True))