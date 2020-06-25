import unittest


class DriverTestCase(unittest.TestCase):
    def test_stack_outputs_and_plot(self):
        from src.examples import stack_outputs_and_plot
        stack_outputs_and_plot()


if __name__ == '__main__':
    unittest.main()
