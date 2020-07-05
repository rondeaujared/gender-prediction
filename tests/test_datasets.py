import unittest
import os
from src.utils import load_config

load_config()


class IMDBTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.imdb_root = os.environ['IMDB_ROOT']

    def test_build_imdb(self):
        from src.datasets import build_imdb
        df = build_imdb(f"{self.imdb_root}/imdb.mat", n=None, save=f"{self.imdb_root}/imdb.pickle")
        print(df)
        print(df[:5])

    def test_load_pickle(self):
        from src.datasets import unpickle_imdb
        df = unpickle_imdb(f"{self.imdb_root}/imdb.pickle")
        print(df)

    def test_imdb_dataset(self):
        from src.datasets import ImdbDataset
        from torchvision.transforms import ToTensor
        from src.datasets import unpickle_imdb
        df = unpickle_imdb(f"{self.imdb_root}/imdb.pickle")
        ds = ImdbDataset(root=self.imdb_root, df=df, transform=ToTensor())

        from torchvision.transforms import ToPILImage
        pil = ToPILImage()
        tensor, label = ds[-1]
        #print(f"Label: {label}")
        #pil(tensor).show()


class VGGFaceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        from torchvision import transforms
        from src.datasets import VGGFaceDataset
        self.root = os.environ['VGGFACE_ROOT']
        self.trans = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
        ])
        self.ds = VGGFaceDataset(self.root, self.trans)

    def test_init(self):
        self.assertEqual(str(self.root), str(self.ds.root))

    def test_nidents(self):
        from src.datasets import VGGFaceDataset
        n = 5
        ds = VGGFaceDataset(self.root, self.trans, nidents=n)
        self.assertEqual(len(ds.identity.keys()), n)

    def test_getitem(self):
        item = self.ds[0]
        self.assertIsNotNone(item)
        item[0].show()

    def test_len(self):
        l_ds = len(self.ds)
        self.assertGreater(l_ds, 0)
        self.assertGreater(l_ds, 10000)

    def test_getiden(self):
        iden = 'n000002'
        expected = [
            '0001_01.jpg',
            '0002_01.jpg',
            '0003_01.jpg'
        ]
        items = self.ds.getiden(iden)
        for exp in expected:
            self.assertIn(exp, items)
        self.assertIsNone(self.ds.getiden('potatosalad'))


if __name__ == '__main__':
    unittest.main()
