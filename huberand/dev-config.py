from daal4py.tests.config import Config

a = Config(
    "/localdisk2/mkl/huberand/scikit-learn-intelex/examples/tests/test_daal4py.yml"
)
b = a["adaboost"]
print(a)
