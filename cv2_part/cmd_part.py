import cv2

from .base_functools import *
from .find_object import *
from .downloload_part import *

def cmd(cmd: bool=True) -> None:
    if cmd:
        while True:
            call_inp = input('> ')
            try:
                if call_inp == 'exit':
                    break
                else:
                    call_func = eval(call_inp)
                if callable(call_func):
                    args = tuple(i for i in eval(input(f'Enter the tuple with args for {call_func.__name__}: ')))
                    core = call_func(*args)
                    if core is None:
                        print('Successfully.')
                        continue
                    elif (tp_core := type(core)) is MatLike:
                        cv2.imshow(f'{call_func.__name__}_n', core)
                    else:
                        print(core)
            except Exception as ex:
                print(f'Error: {ex}.')
    


if __name__ == "__main__":
    cmd(cmd=True)
    # def blur_comparison(img):
    #     cv2.imshow('averaging', blur_img(img, 'averaging', 3))
    #     cv2.imshow('gaussian', blur_img(img, 'gaussian', 3))
    #     cv2.imshow('median', blur_img(img, 'median', 3))
    #     cv2.waitKey(0)

    # info_dict = {}
    # for i in range(1, 5):
    #     way_to_picture = f'./data_set/tokey_{i}.jpg'
    #     img = loader(way_to_picture, print_status=False, method_to_open=cv2.IMREAD_GRAYSCALE) 
    #     info_dict[way_to_picture.split('/')[-1].split('.')[0]] = photo_info(img)
    #     find_contours_of_cards(img, print_status=True)
    # return info_dict
