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

    def test_gender_analyze(self):
        import os
        import pickle
        from src.gender_estimation import gender_analyze
        from src.datasets import ImdbDataset, unpickle_imdb
        from src.utils import load_config
        from src.convnets.utils import IMAGENET_MEAN, IMAGENET_STD
        from torchvision import transforms
        load_config()
        trans = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        imdb_root = os.environ['IMDB_ROOT']
        df = unpickle_imdb(f"{imdb_root}/imdb.pickle")
        ds = ImdbDataset(root=imdb_root, df=df[:250], transform=trans, include_path=True)
        log = gender_analyze(self.weights, ds)
        pickle.dump(log, open('tmp/gender_analyze_log.p', 'wb'))

    def test_cluster(self):
        import pickle
        from src.gender_estimation import cluster
        log = pickle.load(open('tmp/gender_analyze_log.p', 'rb'))
        cluster(log)


if __name__ == '__main__':
    unittest.main()
