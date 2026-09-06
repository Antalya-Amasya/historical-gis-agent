"""G6AV: subsequent campaign is a legacy occurrence boundary."""

from __future__ import annotations

from backend.tests.test_g6ar_legacy_occurrence_boundary_authority import (
    assert_no_rhone_alps_fusion,
    assert_rhone_alps_preserved,
    authority,
)


def test_a_subsequent_campaign_blocks_same_subject_fusion():
    assert_no_rhone_alps_fusion(
        "Ariston crossed the Rhone, then during a subsequent campaign Ariston entered the Alps.",
    )


def test_b_subsequent_campaign_blocks_pronoun_continuation():
    assert_no_rhone_alps_fusion(
        "Ariston crossed the Rhone. During a subsequent campaign he entered the Alps.",
    )


def test_c_subsequent_campaign_blocks_repeated_subject_continuation():
    assert_no_rhone_alps_fusion(
        "Ariston crossed the Rhone. In a subsequent campaign Ariston entered the Alps.",
    )


def test_d_plain_pronoun_continuation_remains_valid():
    assert_rhone_alps_preserved("Ariston crossed the Rhone, then he entered the Alps.")


def test_e_ordinary_campaign_mention_remains_valid():
    assert_rhone_alps_preserved("During the campaign Ariston crossed the Rhone, then he entered the Alps.")


def test_f_named_campaign_continuation_remains_valid():
    assert_rhone_alps_preserved("During Campaign Alpha Ariston crossed the Rhone, then he entered the Alps.")


def test_g_same_campaign_is_not_a_boundary():
    assert authority("Ariston crossed the Rhone. In the same campaign he entered the Alps.") is True


def test_h_existing_another_campaign_boundary_remains_blocked():
    assert_no_rhone_alps_fusion("Ariston crossed the Rhone. In another campaign Ariston entered the Alps.")
