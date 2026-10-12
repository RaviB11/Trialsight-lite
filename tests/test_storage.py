import duckdb
import pandas as pd
import pytest

from trialsight.storage import export_parquet, write_tables


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.duckdb"
    write_tables(path, {
        "studies": pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000002"],
            "start_date": pd.to_datetime(["2012-05-01", None]),
            "n_arms": [2, None],
            "has_us_site": [True, False],
            "why_stopped": [None, None],          # all-null column
        }),
        "empty_table": pd.DataFrame({"nct_id": pd.Series([], dtype=object)}),
    })
    return path


def _read(path):
    with duckdb.connect() as con:
        return con.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def test_export_roundtrip_matches_table(db, tmp_path):
    (p,) = export_parquet(db, ["studies"], tmp_path / "out")
    got = _read(p)
    assert got["nct_id"].tolist() == ["NCT00000001", "NCT00000002"]
    assert got["has_us_site"].tolist() == [True, False]
    assert pd.isna(got.loc[1, "start_date"]) and got.loc[0, "start_date"] == pd.Timestamp("2012-05-01")
    assert got["why_stopped"].isna().all()


def test_export_empty_table(db, tmp_path):
    (p,) = export_parquet(db, ["empty_table"], tmp_path / "out")
    assert len(_read(p)) == 0


def test_path_with_quote_and_space(db, tmp_path):
    (p,) = export_parquet(db, ["studies"], tmp_path / "Robbie's exports")
    assert p.exists() and len(_read(p)) == 2


@pytest.mark.parametrize("name,msg", [("missing", "not found"), ("bad;name", "Unsafe")])
def test_export_rejects_bad_tables(db, tmp_path, name, msg):
    with pytest.raises(ValueError, match=msg):
        export_parquet(db, [name], tmp_path / "out")
