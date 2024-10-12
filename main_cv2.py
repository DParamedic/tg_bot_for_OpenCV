# Надо преобразовать функцию pixel_change() для однопространственных изображений

from downloload_part import loader, downloader
import cv2
from typing import TypeVar, NoReturn, Union
import numpy as np
from math import sqrt

Obj_cv2 = TypeVar('Object_cv2')

def old_model_func(img: Obj_cv2) -> Obj_cv2:
    ...

def new_model_func(img: Obj_cv2=None, *, way_to_picture: str=None) -> Obj_cv2:
    ...

def photo_info(img: Obj_cv2=None, return_status: bool=True, print_status: bool=False, *, way_to_picture: str=None) -> Union[dict, NoReturn]:
    if img is None:
        img = loader(way_to_picture)
    if print_status:
        print("Высота:"+str(img.shape[0]))
        print("Ширина:" + str(img.shape[1]))
        print("Количество каналов:" + str(img.shape[2]))
    if return_status:
        try:
            res_dict = dict(height=img.shape[0], width=img.shape[1], channel_count=img.shape[2])
        except IndexError:
            res_dict = dict(height=img.shape[0], width=img.shape[1], channel_count=1)
        return res_dict

def pixel_info(img: Obj_cv2=None, height_c: int=None, width_c: int=None, *, way_to_picture: str=None) -> tuple:
    if img is None:
        img = loader(way_to_picture)
    if height_c is None:
        height_c = eval(input('Enter height_c: '))
    if width_c is None:
        width_c = eval(input('Enter width_c: '))
    if photo_info(img)['channel_count'] == 1:
        black = img[height_c, width_c]
        color_tuple = (black, )
    else:
        blue, green, red = img[height_c, width_c]
        color_tuple = (blue, green, red)
    return color_tuple

def pixel_change(img: Obj_cv2=None, rgb_tuple: tuple=None, height_c: int=None, width_c: int=None, *, way_to_picture: str=None) -> Obj_cv2:
    if img is None:
        img = loader(way_to_picture)
    if height_c is None:
        height_c = eval(input('Enter height_c: '))
    if width_c is None:
        width_c = eval(input('Enter width_c: '))
    if rgb_tuple is None:
        rgb_tuple = eval(input('Введите кортеж rgb: '))
    img[height_c, width_c] = rgb_tuple
    return img


def resizing(img: Obj_cv2=None, new_wight: int=None, new_height: int=None, *, way_to_picture: str=None) -> Obj_cv2:
    if img is None:
        img = loader(way_to_picture)
    height, wight = img.shape[:2] 
    if new_wight is None and new_height is None:
        print(f'Not change: coordinats is {(new_wight, new_height)}.')
        return img
    if new_wight is None:
        ratio = new_height / height
        new_size_tuple = int(wight * ratio), new_height
    elif new_height is None:
        ratio = new_wight / wight
        new_size_tuple = int(height * ratio), new_wight
    else:
        new_size_tuple = new_height, new_wight

    res_img = cv2.resize(img, new_size_tuple) # resize() меняет разрешение картинки на любое другое.
    # Вторым аргументом может принимать различные методы интрополяции --
    # например, cv2.INTER_NEAREST -- интрополяция методом ближайшего соседа, самый простой и быстрый способ. 
    # При увелечении или сжатии более, чем в 1,5 раз, скорость выполенения находится примерно на одном уровне с другими методами.
    # Так же существуют и ииные методы: cv2.INTER_LINEAR -- по умолчанию, наиболее предпочтителен для увеличения изображения.
    # cv2.INTER_AREA -- наиболее предпочтителен для уменьшения изображения, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4.
    # при изменении размерности важно учитывать соотношение сторон.
    return res_img

def shifting(img: Obj_cv2=None, *, way_to_picture: str=None) -> Obj_cv2:
    if img is None:
        img = loader(way_to_picture)
    height, width = img.shape[:2]
    shift_matrix = np.float32([[1, 0, 200], [0, 1, 300]]) 
    # массив, где в первой строке последняя цифра -- горизантальное смещение, отричательное или положительное (соотв. лево и право),
    # а вторая строка -- соотетственно вверх или вниз.
    shift_img = cv2.warpAffine(img, shift_matrix, (width, height)) # Собственно перемещение.
    return shift_img

def cropping(img: Obj_cv2=None, left_up_point: Union[tuple[int, int], list[int, int]]=None, right_down_point: Union[tuple[int], list[int]]=None, *, way_to_picture: str=None) -> Obj_cv2:
    if img is None:
        img = loader(way_to_picture)
    crop_img = img[left_up_point[0]:right_down_point[0], left_up_point[1]:right_down_point[1]] # [height1 : height2, width1 : width2]
    return crop_img

def rotation(img: Obj_cv2=None, slope: int=45, *, way_to_picture: str=None) -> Obj_cv2:
    if img is None:
        img = loader(way_to_picture)
    (height, width) = img.shape[:2]
    center = (int(width / 2), int(height / 2))
    gradate = round((height if height >= width else width) / sqrt(height**2 + width**2), 1)
    rotation_matrix = cv2.getRotationMatrix2D(center, slope, gradate)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img


def rgb_decomposition(img: Obj_cv2, print_status: bool=False) -> tuple[Obj_cv2]:
    """Return tuple wich blue, green, red in this order."""
    blue, green, red = cv2.split(img)
    if print_status:
        cv2.imshow('blue', blue)
        cv2.imshow('green', green)
        cv2.imshow('red', red)
        cv2.waitKey(0)
    return (blue, green, red)

def rgb_unification(brg_tuple: Union[tuple, list], print_status: bool=False) -> Obj_cv2:
    img = cv2.merge([i for i in  brg_tuple])
    return img

# При сложении: cv2.add обеспечивает сложение и вычитание до конечного значения: до 255 или до 0, даже если предположительно значение выходит за рамки массива.
# Если использовать сложение методом применения знака '+', то значение будет "зацикливаться (255+1=0; 0-1=255).

# В cv2 применяется три основных метода размытия -- averaging(усреднённое), gaussian(гауссово) и median(медианное)
# При использовании averaging создается т.н. усреденная матрица с неким ядром. Далее производится операция т.н. "Свертка" --
# в центре ядра пиксель "усредняется" по значения по сравнению с окружающим. Размерность ядра может быть разной, с увеличением ее растет и коэф. размытости.
# Домыслы: похоже, что при указании большой размерности алгоритм работает примерно так (для 5*5):
# центр усредняется по 3*3, далее каждый из 8, входящих в 3*3, но не в центр также усредняется.
# Метод blur(img, tuple(n*n)=core_size).
# При использовании gaussian вместо простого среднего используется взвешенное среднее. Размытое изображении выглядит более естественно.
# Метод GaussianBlur(img, tuple(n*n)=core_size, m=gaussian_core_deviation(0=auto)).
# В медианном размытии центральный пиксель изображения заменяется медианой всех пикселей в области ядра.
# method medianBlur(img, core_sise: int).
def blur_img(img: Obj_cv2, method: str, compr_ratio: int, gaussian_core_deviation: int=0)-> Obj_cv2:
    if compr_ratio == 1:
        core_tuple = 4, 4
    elif compr_ratio == 2:
        core_tuple = 7, 7
    elif compr_ratio == 3:
        core_tuple = 11, 11
    else:
        raise ValueError('So big!!!!!')
    if method == 'averaging':
        res_img = cv2.blur(img, core_tuple)
        return res_img
    elif method == 'gaussian':
        res_img = cv2.GaussianBlur(img, core_tuple, gaussian_core_deviation)
        return res_img
    elif method == 'median':
        res_img = cv2.medianBlur(img, core_tuple[0])
        return res_img
    else:
        raise TypeError(f'Do not ask method {method}.')
    
def find_contours_of_cards(img) -> Obj_cv2:
    '''Подавать на вход ч/б изображение.'''
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    T, thresh_img = cv2.threshold(blurred, 215, 255, cv2.THRESH_BINARY) 
    # Преобразуем в бинарное изображение. 
    # Первый аргумент -- преобразуемое изображение, второй аргумент -- пороговое значение, начиная от которого все последующие преобразуются до третьего аргумента.
    # cv2.THRESH_BINARY для того, чтобы второй параметр преобразовался ко третьему.
    cv2.imshow('trash_binary', thresh_img)
    cv2.waitKey(0)
    return
    (_, cnts, _) = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # cv2.RETR_EXTERNAL -- только внешние контуры. Для извлечения всех конткров применяется cv2.RETR_LIST.
    # Последний метод метод аппроксимации контура. cv2.CHAIN_APPROX_SIMPLE -- удаление лишних точек в целях экономиии памяти.
    return cnts

def main(cmd: str=None) -> NoReturn:
    def blur_comparison(img):
        cv2.imshow('averaging', blur_img(img, 'averaging', 3))
        cv2.imshow('gaussian', blur_img(img, 'gaussian', 3))
        cv2.imshow('median', blur_img(img, 'median', 3))
        cv2.waitKey(0)

    way_to_picture = './pictures/girl500_500.jpg'
    img = loader(way_to_picture, see_picture=True, method_to_open=cv2.IMREAD_GRAYSCALE)
    print(photo_info(img))
    print(pixel_info(img, 1, 1))
    find_contours_of_cards(img)


if __name__ == "__main__":
    main()