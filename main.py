import myfunctions


# 画像を用意したよ。
target_images = myfunctions.get_all_image_paths()
mat_images = [myfunctions.read_image(_) for _ in target_images]

# 1枚に連結するよ。
# NOTE: 閲覧したければ myfunctions.show_image(concatenated_mat_image);exit()
concatenated_mat_image = myfunctions.concatenate_mat(mat_images, 10)

# Detection を行う。
# 画像から faceId を取得するということ。
detected_faces = myfunctions.detect_with_mat(concatenated_mat_image)
detected_face_ids = [_.face_id for _ in detected_faces]

# Identification を行う。
# faceId に対応する candidate を取得するということ。
# 10件ずつのみ可能。
# faceId を10件ずつに分割。
for face_ids in (detected_face_ids[i:i + 10]
                 for i in range(0, len(detected_face_ids), 10)):
    identify_results = myfunctions.identify(face_ids)
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
