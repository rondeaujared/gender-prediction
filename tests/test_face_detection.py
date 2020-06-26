import unittest


class TestFaceDetection(unittest.TestCase):
    def test_images(self):
        from os import listdir
        from os.path import (join, isfile)
        from src.face_detection import predict_faces, preds_to_txt
        root = f'test_images/'
        files = [join(root, f) for f in listdir(root)
                 if isfile(join(root, f))]
        with open(f'tmp/images.txt', 'w') as f:
            [f.write(f'{item}\n') for item in files]
        l_faces = predict_faces(f'tmp/images.txt')
        preds_to_txt(l_faces, f'tmp/images_face_preds.txt')

    def test_text_to_preds(self):
        import filecmp
        from src.face_detection import txt_to_preds, preds_to_txt
        p1 = f'tmp/images_face_preds.txt'
        p2 = f'tmp/text_to_preds_to_text.txt'
        preds = txt_to_preds(p1)
        preds_to_txt(preds, p2)
        self.assertTrue(filecmp.cmp(p1, p2))

    #def test_draw_faces(self):



if __name__ == '__main__':
    unittest.main()
