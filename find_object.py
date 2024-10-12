from base_functools import *
import cv2
from typing import TypeVar, NoReturn, Union

MatLike = TypeVar('MatLike')

def find_contours_of_cards(img, print_status: bool=False, thrash: int=200) -> MatLike:
    '''B/w image to input.'''
    if photo_info(img)['channel_count'] != 1:
        raise ValueError('No b/w image found.')
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    T, thresh_img = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY) 
    # Преобразуем в бинарное изображение. 
    # Первый аргумент -- преобразуемое изображение, второй аргумент -- пороговое значение, начиная от которого все последующие преобразуются до третьего аргумента.
    # cv2.THRESH_BINARY для того, чтобы второй параметр преобразовался ко третьему.
    if print_status:
        cv2.imshow('trash_binary', thresh_img)
        cv2.waitKey(0)
        return
    (_, cnts, _) = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # cv2.RETR_EXTERNAL -- только внешние контуры. Для извлечения всех конткров применяется cv2.RETR_LIST.
    # Последний метод метод аппроксимации контура. cv2.CHAIN_APPROX_SIMPLE -- удаление лишних точек в целях экономиии памяти.
    return cnts