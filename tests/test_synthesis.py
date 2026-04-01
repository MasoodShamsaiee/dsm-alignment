import pandas as pd


def test_alignment_overview_table_counts_and_shares():
    from dsm_alignment import alignment_overview_table

    results = {
        "flexd": pd.DataFrame({"alignment_class": ["a", "a", "b"]}, index=["H1A", "H1B", "H1C"]),
        "hilo": pd.DataFrame({"alignment_class": ["x", "y"]}, index=["G1A", "G1B"]),
    }
    out = alignment_overview_table(results)
    assert {"program", "alignment_class", "count_fsa", "share_fsa"}.issubset(out.columns)
    assert float(out.loc[(out["program"] == "flexd") & (out["alignment_class"] == "a"), "share_fsa"].iloc[0]) == 2 / 3
