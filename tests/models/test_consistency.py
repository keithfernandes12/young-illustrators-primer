"""Stage 6, character consistency: the per-book score and the threshold experiment.

Everything runs on ``FakeEmbedder`` and in-memory bytes: no weights, no network, no bucket. "Images"
are byte strings whose mix of values stands in for what a character looks like.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from primer.illustration.consistency import ConsistencyScorer, FakeEmbedder, ImageEmbedder, summarise
from primer.illustration.consistency import experiment as exp


def picture(*values: int, size: int = 64) -> bytes:
    """An 'image' made of the given byte values, repeated."""
    return bytes(values[index % len(values)] for index in range(size))


CAT = picture(1, 2, 3)
CAT_AGAIN = picture(1, 2, 3, 3)  # close to CAT, not identical
DOG = picture(9, 10, 11)


# ------------------------------------------------------------------ the score


def test_pages_of_the_reference_character_score_high():
    result = ConsistencyScorer(FakeEmbedder()).score([CAT], [CAT, CAT_AGAIN, CAT])
    assert result.score > 0.9
    assert result.cohesion is not None and result.cohesion > 0.9


def test_identical_pages_of_the_wrong_character_score_low():
    """The failure the metric exists for: pages agree with each other but not with the drawing."""
    result = ConsistencyScorer(FakeEmbedder()).score([CAT], [DOG, DOG, DOG])
    assert result.cohesion == 1.0
    assert result.score < 0.5


def test_no_reference_or_no_pages_scores_zero():
    scorer = ConsistencyScorer(FakeEmbedder())
    assert scorer.score([], [CAT, CAT]).score == 0.0
    assert scorer.score([], [CAT, CAT]).identity == (0.0, 0.0)
    assert scorer.score([CAT], []).score == 0.0


def test_a_single_page_is_scored_on_identity_alone():
    result = ConsistencyScorer(FakeEmbedder()).score([CAT], [CAT])
    assert result.cohesion is None
    assert result.score == 1.0


def test_each_page_is_held_to_its_closest_reference():
    """A book with two characters: a page needs to look like one of them, not both."""
    result = ConsistencyScorer(FakeEmbedder()).score([CAT, DOG], [CAT, DOG])
    assert result.identity == (1.0, 1.0)


def test_weakest_page_names_the_off_model_page():
    result = ConsistencyScorer(FakeEmbedder()).score([CAT], [CAT, DOG, CAT_AGAIN])
    assert result.weakest_page == 1


def test_weighting_in_cohesion_lifts_a_wrong_character_book():
    """Why identity is the default score: cohesion rewards pages that agree on the wrong character."""
    identity_only = ConsistencyScorer(FakeEmbedder()).score([CAT], [DOG, DOG]).score
    blended = ConsistencyScorer(FakeEmbedder(), identity_weight=0.5).score([CAT], [DOG, DOG]).score
    assert blended > identity_only


def test_min_aggregate_is_stricter_than_mean():
    pages = [CAT, CAT, DOG]
    mean = ConsistencyScorer(FakeEmbedder(), identity_weight=1.0).score([CAT], pages).score
    strict = ConsistencyScorer(FakeEmbedder(), identity_weight=1.0, aggregate="min").score([CAT], pages).score
    assert strict < mean


def test_threshold_decides_consistent_and_none_means_report_only():
    assert ConsistencyScorer(FakeEmbedder()).score([CAT], [CAT]).consistent is None
    assert ConsistencyScorer(FakeEmbedder(), threshold=0.8).score([CAT], [CAT]).consistent is True
    assert ConsistencyScorer(FakeEmbedder(), threshold=0.8).score([CAT], [DOG]).consistent is False


def test_a_book_is_embedded_in_one_batch():
    """It runs in the request path: one model call per book, not one per page."""
    embedder = FakeEmbedder()
    ConsistencyScorer(embedder).score([CAT], [CAT, CAT_AGAIN, DOG, CAT])
    assert embedder.calls == 1


@pytest.mark.parametrize("options", [{"identity_weight": 1.5}, {"aggregate": "median"}])
def test_bad_settings_are_refused(options):
    with pytest.raises(ValueError):
        ConsistencyScorer(FakeEmbedder(), **options)


def test_summarise_gives_the_gated_key_and_the_pass_rate():
    scorer = ConsistencyScorer(FakeEmbedder(), threshold=0.8)
    results = [scorer.score([CAT], [CAT]), scorer.score([CAT], [DOG])]
    summary = summarise(results)
    assert summary["clip_consistency"] == round((results[0].score + results[1].score) / 2, 4)
    assert summary["consistency_pass_rate"] == 0.5
    assert "consistency_pass_rate" not in summarise([ConsistencyScorer(FakeEmbedder()).score([CAT], [CAT])])
    assert summarise([]) == {"clip_consistency": 0.0}


def test_fake_embedder_satisfies_the_protocol():
    assert isinstance(FakeEmbedder(), ImageEmbedder)


def test_importing_the_area_loads_no_model_libraries_or_data_pipeline():
    """Serving imports this per request path; weights and primer.data must stay out of it."""
    code = ("import sys, primer.illustration.consistency, primer.illustration.consistency.embedders; "
            "print(sorted(m for m in ('torch', 'transformers', 'PIL', 'primer.data') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ------------------------------------------------------------------ the experiment


def books(n_books: int = 40, pages_per_book: int = 4, sources: tuple[str, ...] = ("asb", "storyweaver")):
    """Picture books where every page of a book shares that book's byte palette, plus a little noise."""
    rng = np.random.default_rng(0)
    rows, images = [], {}
    for book in range(n_books):
        palette = rng.choice(256, size=3, replace=False)
        for page in range(pages_per_book):
            uri = f"memory://book{book}/page{page}.png"
            noise = rng.integers(256, size=8)
            images[uri] = picture(*palette, size=96) + bytes(int(value) for value in noise)
            rows.append({"source": sources[book % len(sources)], "book_id": str(book), "page_number": page + 1,
                         "language": "English" if book % 4 else "Kiswahili", "image_uri": uri})
    # A page with no illustration, and a book with only one, are both unusable for pairs.
    rows.append({"source": "asb", "book_id": "0", "page_number": 99, "language": "English", "image_uri": None})
    rows.append({"source": "asb", "book_id": "lonely", "page_number": 1, "language": "English",
                 "image_uri": "memory://lonely.png"})
    images["memory://lonely.png"] = picture(1)
    return pd.DataFrame(rows), images


def config(**overrides):
    base = exp.ExperimentConfig(name="test", data_version="v-test", baseline="weak",
                                candidates=("weak", "strong"), pairs_per_split=200, seed=1, batch_size=16)
    return exp.ExperimentConfig(**{**base.__dict__, **overrides})


class WeakEmbedder(FakeEmbedder):
    """Two buckets only: every image looks much the same, so it barely separates books."""

    name = "weak"

    def __init__(self) -> None:
        super().__init__(dimensions=2)


def test_split_is_by_book_and_stable():
    pages, _ = books()
    illustrated = exp.illustrated_pages(pages, val_fraction=0.5)
    assert (illustrated.groupby("book_key")["split"].nunique() == 1).all()
    assert exp.split_of("asb/7", 0.5) == exp.split_of("asb/7", 0.5)
    assert set(illustrated["split"]) == {"val", "test"}


def test_unillustrated_pages_and_single_page_books_are_dropped():
    pages, _ = books()
    illustrated = exp.illustrated_pages(pages, val_fraction=0.5)
    assert illustrated["image_uri"].notna().all()
    assert "asb/lonely" not in set(illustrated["book_key"])


def test_book_ids_are_only_unique_within_a_source():
    pages = pd.DataFrame([
        {"source": "asb", "book_id": "1", "language": "English", "image_uri": "memory://a1"},
        {"source": "storyweaver", "book_id": "1", "language": "English", "image_uri": "memory://s1"},
    ])
    # Two different books, one page each: neither has two pages, so neither survives.
    assert exp.illustrated_pages(pages, 0.5).empty


def test_pairs_are_labelled_by_book_and_negatives_stay_within_a_source():
    pages, _ = books()
    illustrated = exp.illustrated_pages(pages, val_fraction=0.5)
    pairs = exp.build_pairs(illustrated, n_pairs=200, seed=3)
    positives, negatives = pairs[pairs["label"] == 1], pairs[pairs["label"] == 0]
    assert (positives["book_a"] == positives["book_b"]).all()
    assert (positives["a"] != positives["b"]).all()
    assert (negatives["book_a"] != negatives["book_b"]).all()
    source_of = illustrated.set_index("book_key")["source"].to_dict()
    assert (negatives["book_b"].map(source_of) == negatives["source"]).all()
    assert 0.4 < pairs["label"].mean() < 0.6
    pd.testing.assert_frame_equal(pairs, exp.build_pairs(illustrated, n_pairs=200, seed=3))


def test_a_book_with_no_language_is_sliced_as_unknown_not_dropped():
    pages, _ = books()
    pages.loc[pages["book_id"] == "1", "language"] = None
    illustrated = exp.illustrated_pages(pages, val_fraction=0.5)
    split = illustrated.loc[illustrated["book_id"] == "1", "split"].iloc[0]
    pairs = exp.build_pairs(illustrated[illustrated["split"] == split], n_pairs=400, seed=2)
    assert pairs["language"].notna().all()
    assert set(pairs.loc[pairs["book_a"] == "storyweaver/1", "language"]) == {"unknown"}
    rows = exp.slice_metrics(pairs, np.zeros(len(pairs)), 0.5, ["language"])
    assert sum(row["n"] for row in rows) == len(pairs)


def test_cached_reader_fetches_each_image_once(tmp_path):
    fetched: list[str] = []

    def fetch(uri: str) -> bytes:
        fetched.append(uri)
        return uri.encode()

    read = exp.CachedReader(tmp_path / "images", read=fetch)
    assert [read("memory://a"), read("memory://a"), read("memory://b")] == [b"memory://a", b"memory://a",
                                                                            b"memory://b"]
    assert fetched == ["memory://a", "memory://b"]
    # A second run over the same folder reads from disk only.
    assert exp.CachedReader(tmp_path / "images", read=fetch)("memory://a") == b"memory://a"
    assert fetched == ["memory://a", "memory://b"]
    assert not list((tmp_path / "images").glob("*.partial"))


def test_pairs_need_two_books():
    pages = pd.DataFrame([{"source": "asb", "book_key": "asb/1", "language": "English", "image_uri": u}
                          for u in ("memory://1", "memory://2")])
    with pytest.raises(ValueError, match="two illustrated books"):
        exp.build_pairs(pages, n_pairs=10, seed=0)


def test_threshold_separates_separable_scores():
    similarity = np.array([0.2, 0.3, 0.35, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    threshold = exp.choose_threshold(similarity, labels)
    assert 0.35 < threshold <= 0.7
    metrics = exp.classification(similarity, labels, threshold)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == metrics["recall"] == metrics["accuracy"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_classification_without_both_classes_has_no_auc():
    assert exp.classification(np.array([0.9, 0.8]), np.array([1, 1]), 0.5)["roc_auc"] is None


def test_small_slices_are_flagged_and_left_out_of_the_gap():
    pairs = pd.DataFrame({"label": [1, 0] * 110 + [1, 0] * 5,
                          "source": ["big"] * 220 + ["tiny"] * 10,
                          "language": ["English"] * 110 + ["Kiswahili"] * 120})
    similarity = np.where(pairs["label"] == 1, 0.9, 0.1).astype(float)
    similarity[-10:] = 0.5  # the tiny slice is unseparable, but too small to judge
    slices = exp.slice_metrics(pairs, similarity, 0.5, ["source", "language"])
    by_value = {(row["key"], row["value"]): row for row in slices}
    assert by_value[("source", "tiny")]["too_small"] is True
    assert by_value[("source", "big")]["too_small"] is False
    gaps = exp.slice_gaps(slices)
    assert "source" not in gaps  # only one judged source slice
    assert gaps["language"]["worst"] == "Kiswahili"


def test_threshold_sweep_trades_recall_for_precision():
    similarity = np.array([0.55, 0.65, 0.75, 0.6, 0.85, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    sweep = exp.threshold_sweep(similarity, labels, thresholds=(0.5, 0.9))
    assert sweep[0]["recall"] == 1.0 and sweep[1]["recall"] < 1.0
    assert sweep[1]["precision"] >= sweep[0]["precision"]


def test_run_selects_the_better_embedder_on_val_and_reports_on_test():
    pages, images = books()
    embedders = {"weak": WeakEmbedder(), "strong": FakeEmbedder(dimensions=256)}
    report = exp.run(config(), pages, embedders, images.__getitem__,
                     perturbations={"reverse": lambda image: image[::-1]})
    assert report["selected"] == "strong"
    assert report["candidates"]["strong"]["val"]["roc_auc"] > report["candidates"]["weak"]["val"]["roc_auc"]
    assert report["selected_test"] == report["candidates"]["strong"]["test"]
    assert report["selected_test"]["roc_auc"] > 0.9
    assert report["threshold"] == report["candidates"]["strong"]["threshold"]
    assert {row["key"] for row in report["slices"]} == {"source", "language"}
    # Reversing bytes leaves a byte histogram unchanged: the fake keys on palette, not order.
    assert report["sensitivity"]["reverse"]["delta"] == 0.0
    assert report["pairs"]["val"]["n"] > 0 and report["pairs"]["test"]["n"] > 0


def test_run_keeps_the_baseline_unless_a_candidate_beats_it():
    pages, images = books()
    same = {"weak": FakeEmbedder(dimensions=256), "strong": FakeEmbedder(dimensions=256)}
    report = exp.run(config(), pages, same, images.__getitem__, perturbations={})
    assert report["selected"] == "weak"  # a tie keeps the baseline


def test_sensitivity_sees_a_perturbation_that_destroys_the_signal():
    pages, images = books()
    report = exp.run(config(candidates=("strong",), baseline="strong"), pages,
                     {"strong": FakeEmbedder(dimensions=256)}, images.__getitem__,
                     perturbations={"blank": lambda image: bytes(len(image))})
    assert report["sensitivity"]["blank"]["delta"] < -0.3


def test_shipped_config_loads_and_names_its_baseline():
    loaded = exp.load_config()
    assert loaded.baseline in loaded.candidates
    assert loaded.data_version == "v1.1"
    assert set(loaded.perturbations) <= set(exp.PERTURBATIONS)


def test_config_must_name_a_baseline_among_its_candidates(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\ndata_version: v1\nbaseline: nope\ncandidates: [a]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="baseline"):
        exp.load_config(path)
