from data_processing import load_cmapss

def test_load_data():
    df = load_cmapss("data/train_FD003.txt")
    assert df is not None
    assert df.shape[1] == 26
