import cv2
import numpy


def read_image(image_path: str) -> numpy.ndarray:

    # 画像を読み込みます。
    # NOTE: imread により画像が mat 形式のデータになります。
    # NOTE: mat ってのは numpy の1,2次元配列です。
    # NOTE: type(mat) -> <class 'numpy.ndarray'>
    # NOTE: 1次元のことを channel といい、2次元のことを dimension といいます。
    mat = cv2.imread(image_path)

    # NOTE: 'not mat' とか書くと The truth value of an array with
    # NOTE: more than one element is ambiguous. と言われる。
    assert mat is not None, f'The image {image_path} cannot be read'

    # cv2.IMREAD_GRAYSCALE を指定すると白黒になる。
    # mat_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # assert mat_grayscale is not None, 'グレースケール画像が読み込めなかったよ。'

    return mat


def show_image(mat_file: numpy.ndarray) -> None:

    # imshow と waitKey のセットで読み込んだ画像が閲覧できます。ニュッと GUI で出てくる。
    cv2.imshow('mat', mat_file)
    cv2.waitKey(0)


def get_image_size(mat_file: numpy.ndarray) -> tuple:

    # channel はカラー画像のとき存在します。
    # NOTE: グレースケールと見分けるのに使われるようだ。
    width, height, channel = mat_file.shape
    return (width, height)


def concatenate_tile(list_2d: list) -> numpy.ndarray:

    # mat_file の1次元配列を受け取り、タイル状に連結します。
    return cv2.vconcat([cv2.hconcat(list_1d) for list_1d in list_2d])


def convert_list_4x4(list_1d: list, blank: object = None) -> list:
    """1次元リストを4x4の2次元リストに変換します。

    Args:
        list_1d (list): 1次元リスト。
        blank (object): 空きスペースに置くオブジェクト。

    Returns:
        list: 2次元リスト。
    """

    if blank is None:
        blank = numpy.ones((100, 100, 3), numpy.uint8) * 255

    list_2d = [[] for i in range(4)]
    i = 0
    for v in range(4):
        for h in range(4):
            if i < len(list_1d):
                list_2d[v].append(list_1d[i])
                i += 1
            else:
                list_2d[v].append(blank)
    return list_2d


def concatenate_mat_4x4(list_1d: list) -> numpy.ndarray:
    """画像の一覧を連結し4x4の mat 形式で取得します。

    Args:
        list_1d (list): mat 形式の画像のリスト。

    Returns:
        numpy.ndarray: 連結したひとつの mat 画像。
    """

    # 画像が64枚に満たないときのための空白画像です。
    blank_mat = numpy.ones((100, 100, 3), numpy.uint8) * 255

    # 4x4の2次元配列に変換します。
    list_2d = convert_list_4x4(list_1d, blank_mat)

    # mat の1次元配列を受け取り、タイル状に連結します。
    return cv2.vconcat([cv2.hconcat(list_1d) for list_1d in list_2d])
