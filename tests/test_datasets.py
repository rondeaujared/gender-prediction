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


if __name__ == '__main__':
    unittest.main()
