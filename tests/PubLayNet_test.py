import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "PubLayNet.py"


@pytest.mark.parametrize(
    argnames="decode_rle",
    argvalues=(
        False,
        True,
    ),
)
def test_load_dataset(dataset_path: str, decode_rle: bool):
    dataset = ds.load_dataset(path=dataset_path, decode_rle=decode_rle)
