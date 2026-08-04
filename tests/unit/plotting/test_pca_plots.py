"""Tests for PCA plotting helpers."""

import pandas as pd
import pytest

from webapp.utilities.plotting.pca_plots import pca_contributions


def test_pca_contributions_matches_reference_values():
    dtm = pd.DataFrame(
        {
            "doc_id": ["doc1", "doc2", "doc3", "doc4"],
            "TagA": [1.0, 2.0, 3.0, 4.0],
            "TagB": [4.0, 2.0, 1.0, 0.0],
            "TagC": [0.5, 1.5, 3.0, 3.5],
        }
    )

    pca_df, contrib_df, variance = pca_contributions(dtm, ["A", "B", "A", "B"])

    assert pca_df["PC1"].tolist() == pytest.approx(
        [3.1495726352741533, 0.758960091525271, -1.2297251776482652, -2.6788075491511587]
    )
    assert pca_df["PC2"].tolist() == pytest.approx(
        [0.16566422146324572, -0.35308644151685975, 0.1713326186599929, 0.01608960139362119]
    )
    assert pca_df["PC3"].tolist() == pytest.approx(
        [
            0.0766327662243805,
            -0.04934110141569103,
            -0.19117039844351422,
            0.16387873363482452,
        ]
    )
    assert pca_df["Group"].tolist() == ["A", "B", "A", "B"]
    assert pca_df["doc_id"].tolist() == ["doc1", "doc2", "doc3", "doc4"]

    assert contrib_df["PC1"].tolist() == pytest.approx(
        [-25.76092349705215, 45.121947785492495, -29.117128717455348]
    )
    assert contrib_df["PC2"].tolist() == pytest.approx(
        [4.337415546231776, 49.468530502324285, 46.194053951443934]
    )
    assert contrib_df["PC3"].tolist() == pytest.approx(
        [69.90166095671606, 5.409521712183215, -24.68881733110071]
    )
    assert contrib_df["Tag"].tolist() == ["TagA", "TagB", "TagC"]
    assert variance == pytest.approx(
        [0.9869613943833889, 0.00934937162422936, 0.0036892339923817277]
    )
