import unittest


class IMDBTestCase(unittest.TestCase):
    def setUp(self) -> None:
        import os
        from src.utils import load_config
        load_config()
        self.imdb_root = os.environ['IMDB_ROOT']

    def test_build_imdb(self):
        from src.datasets import build_imdb
        df = build_imdb(f"{self.imdb_root}/imdb.mat", n=1000, save=f"{self.imdb_root}/imdb.pickle")
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


if __name__ == '__main__':
    unittest.main()
