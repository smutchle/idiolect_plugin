import fidelity as fd


SRC = ("In 2021, researchers at Stanford University recruited 120 participants over 3 weeks. "
       "Accuracy rose by 4.2% (p < 0.05) relative to the baseline [3], consistent with Smith et al. (2020). "
       "The effect may generalize to other settings.")


def test_identical_text_passes():
    r = fd.compare_texts(SRC, SRC)
    assert r["n_issues"] == 0


def test_detects_number_unit_citation_entity_changes():
    out = ("In 2021, researchers at Stanford recruited 100 participants over three weeks. "
           "Accuracy rose by 4.5% relative to the baseline, as Jones (2019) found at Harvard. "
           "The effect may generalize to other settings.")
    r = fd.compare_texts(SRC, out)
    assert "120" in r["numbers"]["missing"] and "100" in r["numbers"]["added"]
    assert "4.2%" in r["numbers"]["missing"] and "4.5%" in r["numbers"]["added"]
    assert "3" not in r["numbers"]["missing"]                      # 3 weeks == three weeks
    assert "120 participant" in r["units"]["missing"] or "120 participants" in r["units"]["missing"]
    assert "3" in r["citations"]["missing"] and "Smith 2020" in r["citations"]["missing"]
    assert "Jones 2019" in r["citations"]["added"]
    assert "Harvard" in r["entities"]["added"]
    assert any("Stanford University" in e for e in r["entities"]["missing"])


def test_number_words_and_formatting_are_normalized():
    a = "It cost $2,000 and took two years; 50 percent of users agreed."
    b = "It took 2 years and cost $2000; 50% of users agreed."
    r = fd.compare_texts(a, b)
    assert r["numbers"]["missing"] == [] and r["numbers"]["added"] == []
    assert r["units"]["missing"] == [] and r["units"]["added"] == []


def test_no_false_positives_on_style_changes():
    a = "One of the most important factors is communication. However, teams must be intentional."
    b = "Communication is one of the most important things; but teams must take deliberate care, and not small ones."
    r = fd.compare_texts(a, b)
    assert r["numbers"]["added"] == [] and r["units"]["added"] == []
    assert r["entities"]["added"] == [] and r["entities"]["missing"] == []


def test_imported_exemplar_name_is_caught():
    a = "Floods in the city caused real inconvenience for commuters."
    b = "The floods in Battersea caused real inconvenience for commuters."
    r = fd.compare_texts(a, b)
    assert "Battersea" in r["entities"]["added"]


def test_formatting_labels_and_name_lists_are_not_entities():
    a = "- 💡 **Speed:** Code generation is faster.\n\nIt was covered by The New York Times, Wired, and The Verge."
    b = "- Code generation is faster.\n\nThe New York Times, Wired and The Verge covered it."
    r = fd.compare_texts(a, b)
    assert r["entities"]["missing"] == [] and r["entities"]["added"] == [], r["entities"]
