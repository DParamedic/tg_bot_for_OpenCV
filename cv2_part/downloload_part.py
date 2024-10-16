from typing import TypeVar, NoReturn

import cv2

MatLike = TypeVar('MatLike')

def loader(way_to_picture: str, print_status: bool=False, *, method_to_open=None, window_name: str=None) -> MatLike:
    img = cv2.imread(way_to_picture, method_to_open) 
    if print_status:
        if window_name is None:
            window_name = way_to_picture.split('/')[-1].split('.')[0]
        print('Press any keybord key to close picture.')
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
    return img

def downloader(cv2_obj: MatLike=None, new_picture_name: str=None) -> NoReturn:
    if new_picture_name is None:
        new_picture_name = input('Enter new picture name (\'picture\' by default): ')
        if new_picture_name == '':
            new_picture_name = 'picture'
    cv2.imwrite(f'{new_picture_name}', cv2_obj)
    

