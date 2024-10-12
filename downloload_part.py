import cv2
from typing import TypeVar, NoReturn

Obj_cv2 = TypeVar('Object_cv2')

def loader(way_to_picture: str, see_picture: bool=False, *, method_to_open=None, window_name: str=None) -> Obj_cv2:
    img = cv2.imread(way_to_picture, method_to_open) 
    if see_picture:
        if window_name is None:
            window_name = way_to_picture.split('/')[-1].split('.')[0]
        print('Press any keybord key to close picture.')
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
    return img

def downloader(cv2_obj: Obj_cv2=None, new_picture_name: str=None, new_picture_expansion: str=None, way_to_save: str='./') -> NoReturn:
    """Use way_to_picture or cv2_obj. If use both this ways, only the cv2_obj will be processing.
    By default new_picture_expansion is 'jpg', new_picture_name is 'picture'."""
    if new_picture_name is None:
        new_picture_name = input('Enter new picture name: ')
        if new_picture_name == '':
            new_picture_name = 'picture'
    if new_picture_expansion is None:
        new_picture_expansion = input('Enter new picture expansion: ')
        if new_picture_expansion == '':
            new_picture_expansion = 'jpg'
    cv2.imwrite(f'{new_picture_name}.{new_picture_expansion}', cv2_obj)
    

