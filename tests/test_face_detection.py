import unittest


class TestFaceDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.root = f'test_images/'
        self.images = f'tmp/images.txt'
        self.face_preds = f'tmp/images_face_preds.txt'

    def test_images(self):
        from src.face_detection import predict_faces, preds_to_txt
        from src.utils import get_files_in_dir
        files = get_files_in_dir(self.root)
        with open(self.images, 'w') as f:
            [f.write(f'{item}\n') for item in files]
        l_faces = predict_faces(self.images)
        preds_to_txt(l_faces, self.face_preds)

    def test_text_to_preds(self):
        import filecmp
        from src.face_detection import txt_to_preds, preds_to_txt
        p1 = self.face_preds
        p2 = f'tmp/text_to_preds_to_text.txt'
        preds = txt_to_preds(p1)
        preds_to_txt(preds, p2)
        self.assertTrue(filecmp.cmp(p1, p2))

    def test_draw_faces(self):
        from src.face_detection import txt_to_preds, draw_faces
        preds = txt_to_preds(self.face_preds)
        face_preds = draw_faces(preds, f'tmp/drawn_faces')


if __name__ == '__main__':
    unittest.main()
