import unittest


class TestGenderEstimation(unittest.TestCase):
    def setUp(self) -> None:
        res = f'test_resources'
        self.images = f'{res}/images.txt'
        self.face_preds = f'{res}/images_face_preds.txt'
        self.gender_preds = f'tmp/images_gender_preds.txt'
        self.weights = f'/mnt/fastdata/SavedModels/resnet18_gender.pth'

    def test_faces_rdy(self):
        from src.face_detection import txt_to_preds
        l_faces = txt_to_preds(self.face_preds)
        self.assertTrue(len(l_faces) > 0)
        return l_faces

    def test_predict_gender(self):
        from src.face_detection import txt_to_preds
        from src.gender_estimation import gender_predict, gender_preds_to_txt
        l_faces = txt_to_preds(self.face_preds)
        path_face = [(path, dict(face)) for path, flist in l_faces for face in flist]
        l_genders = gender_predict(path_face, self.weights)
        gender_preds_to_txt(l_genders, self.gender_preds)

    def test_draw_genders(self):
        from src.gender_estimation import gender_txt_to_preds, draw_genders
        l_genders = gender_txt_to_preds(self.gender_preds)
        draw_genders(l_genders, 'tmp/drawn_genders')


if __name__ == '__main__':
    unittest.main()
