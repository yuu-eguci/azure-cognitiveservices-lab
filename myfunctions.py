# Built-in modules.
import os
import io

# Third-party modules.
import cv2
import numpy
import dotenv
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import GroupResult
from msrest.authentication import CognitiveServicesCredentials


# 今回のリポジトリはラボです。 .env を使うことにします。
dotenv.load_dotenv(dotenv.find_dotenv(raise_error_if_not_found=True))

FACE_ENDPOINT = os.environ['FACE_ENDPOINT']
FACE_SUBSCRIPTION_KEY = os.environ['FACE_SUBSCRIPTION_KEY']
PERSON_GROUP_ID = os.environ['PERSON_GROUP_ID']


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


def convert_list_2d(list_1d: list, size: int, blank: object = None) -> list:
    """1次元リストを正方形2次元リストに変換します。

    Args:
        list_1d (list): 1次元リスト。
        size (int): 作成するリスト1辺の長さ。
        blank (object): 空きスペースに置くオブジェクト。

    Returns:
        list: 2次元リスト。
    """

    if blank is None:
        blank = numpy.ones((100, 100, 3), numpy.uint8) * 255

    list_2d = [[] for i in range(size)]
    i = 0
    for v in range(size):
        for h in range(size):
            if i < len(list_1d):
                list_2d[v].append(list_1d[i])
                i += 1
            else:
                list_2d[v].append(blank)
    return list_2d


def concatenate_mat(list_1d: list, size: int) -> numpy.ndarray:
    """画像の一覧を連結し size x size の mat 形式で取得します。

    Args:
        list_1d (list): mat 形式の画像のリスト。
        size (int): 作成する画像1辺の長さ。

    Returns:
        numpy.ndarray: 連結したひとつの mat 画像。
    """

    # 画像が size x size 枚に満たないときのための空白画像です。
    blank_mat = numpy.ones((100, 100, 3), numpy.uint8) * 255

    # 指定したサイズの2次元配列に変換します。
    list_2d = convert_list_2d(list_1d, size, blank_mat)

    # mat の1次元配列を受け取り、タイル状に連結します。
    return cv2.vconcat([cv2.hconcat(list_1d) for list_1d in list_2d])


def convert_mat2stream(mat: numpy.ndarray) -> io.BytesIO:
    """Mat を stream に変換します。 Python で stream といえば io.BytesIO らしい。"""

    encode_succeeded, buffer = cv2.imencode('.png', mat)
    stream = io.BytesIO(buffer)
    return stream


def create_face_client() -> FaceClient:

    return FaceClient(
        FACE_ENDPOINT,
        CognitiveServicesCredentials(FACE_SUBSCRIPTION_KEY))


def detect_with_mat(mat: numpy.ndarray) -> list:
    """画像を送って、 detection の結果を DetectedFace """

    face_client = create_face_client()
    stream = convert_mat2stream(mat)

    # ドキュメント: detect_with_stream
    # https://docs.microsoft.com/ja-jp/python/api/azure-cognitiveservices-vision-face/azure.cognitiveservices.vision.face.operations.faceoperations?view=azure-python
    detected_faces = face_client.face.detect_with_stream(
        stream, recognition_model='recognition_02')
    # ドキュメント: DetectedFace
    # https://docs.microsoft.com/ja-jp/python/api/azure-cognitiveservices-vision-face/azure.cognitiveservices.vision.face.models.detectedface
    return detected_faces


def identify(face_ids: list) -> list:

    face_client = create_face_client()

    # ドキュメント: IdentifyResult
    # https://docs.microsoft.com/ja-jp/python/api/azure-cognitiveservices-vision-face/azure.cognitiveservices.vision.face.models.identifyresult
    identify_results = face_client.face.identify(
        face_ids,
        person_group_id=PERSON_GROUP_ID,
        max_num_of_candidates_returned=1,
        confidence_threshold=.65)
    return identify_results


def group(face_ids: list) -> GroupResult:

    face_client = create_face_client()

    # ドキュメント: GroupResult
    # https://docs.microsoft.com/ja-jp/python/api/azure-cognitiveservices-vision-face/azure.cognitiveservices.vision.face.operations.faceoperations
    group_result = face_client.face.group(face_ids)
    return group_result
