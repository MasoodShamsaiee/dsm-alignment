import pandas as pd


def test_package_imports():
    import dsm_alignment
    from dsm_alignment import evaluate_flexd_alignment, run_all_program_alignments

    assert hasattr(dsm_alignment, "city_fsa_feature_table")
    assert callable(evaluate_flexd_alignment)
    assert callable(run_all_program_alignments)


def test_basic_alignment_smoke():
    from dsm_alignment import evaluate_flexd_alignment

    df = pd.DataFrame(
        {
            "Labour force status / Worked full-year, full-time in 2020 / 25% sample data": [0.6, 0.2, 0.4],
            "Labour force status / Not in the labour force / 25% sample data": [0.2, 0.5, 0.3],
            "Census families / One-parent census families / %": [0.1, 0.2, 0.05],
            "Commuting duration / 60 minutes and over / %": [0.3, 0.1, 0.2],
            "Occupied private dwellings / Average household size": [2.4, 1.8, 3.1],
            "Occupied private dwellings / Average number of persons per room": [0.5, 0.7, 0.6],
            "Tenure / Owner": [0.7, 0.4, 0.6],
            "peak_load": [10.0, 8.0, 12.0],
            "winter_peak_share": [0.4, 0.2, 0.4],
            "winter_peak_intensity": [1.3, 1.1, 1.5],
            "heating_slope_per_hdd": [0.8, 0.4, 1.0],
            "am_pm_peak_ratio": [1.2, 0.9, 1.4],
        },
        index=["H1A", "H1B", "H1C"],
    )

    out = evaluate_flexd_alignment(df)
    assert "alignment_class" in out.columns
    assert len(out) == 3
