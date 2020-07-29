import myfunctions


# 画像を27枚用意したよ。
target_images = [
    './images/100x100-dog.png',
    './images/100x100-egc.png',
    './images/100x100-egc2.png',
    './images/100x100-kbt.png',
    './images/100x100-ymzk.png',
    './images/100x100-01.png',
    './images/100x100-02.png',
    './images/100x100-03.png',
    './images/100x100-04.png',
    './images/100x100-05.png',
    './images/100x100-06.png',
    './images/100x100-07.png',
    './images/100x100-08.png',
    './images/100x100-09.png',
    './images/100x100-10.png',
    './images/100x100-11.png',
    './images/100x100-12.png',
    './images/100x100-13.png',
    './images/100x100-14.png',
    './images/100x100-15.png',
    './images/100x100-16.png',
    './images/100x100-17.png',
    './images/100x100-18.png',
    './images/100x100-19.png',
    './images/100x100-20.png',
    './images/100x100-21.png',
    './images/100x100-22.png',
]
mat_images = [myfunctions.read_image(_) for _ in target_images]

# 1枚に連結するよ。
# NOTE: 閲覧したければ myfunctions.show_image(concatenated_mat_image)
concatenated_mat_image = myfunctions.concatenate_mat(mat_images, 6)

# Detection を行う。
# 画像から faceId を取得するということ。
detected_faces = myfunctions.detect_with_mat(concatenated_mat_image)
detected_face_ids = [_.face_id for _ in detected_faces]

# Identification を行う。
# faceId に対応する candidate を取得するということ。
identify_results = myfunctions.identify(detected_face_ids)
for result in identify_results:
    print('face_id:', result.face_id)
    if result.candidates:
        print('candidate.person_id:', result.candidates[0].person_id)
        print('candidate.confidence:', result.candidates[0].confidence)
    else:
        print('No candidates')

# Grouping を行う。
# faceId のリストを類似性をもとに分別する。
group_result = myfunctions.group(detected_face_ids)
for group in group_result.groups:
    print('group:', group)
print('messy_group:', group_result.messy_group)
