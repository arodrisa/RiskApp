import pandas as pd

from src.dashboard import merge_benchmark


def test_merge_benchmark(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame({"Date": ["2021-01-01", "2021-01-02"], "Adj Close": [100, 101]})
    file = data_dir / "SPY.csv"
    df.to_csv(file, index=False)

    result = merge_benchmark("SPY", data_dir=str(data_dir))
    assert list(result) == [100, 101]
