import unittest


class DriverTestCase(unittest.TestCase):
    def test_stack_outputs_and_plot(self):
        from src.examples import stack_outputs_and_plot
        stack_outputs_and_plot()

    def test_gender_estimation(self):
        from src.examples import gender_estimation
        gender_estimation()


if __name__ == '__main__':
    unittest.main()
