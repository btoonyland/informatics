import pytest

from informatics.cleaning import is_non_english, is_withdrawn, near_duplicates

ENGLISH = (
    "We propose a deep learning model for tumor segmentation in brain MRI scans and evaluate it "
    "on three hospital cohorts, where it improves the Dice score over strong baselines."
)
TECHNICAL = (
    "We propose U-Net++ with ResNet-50 encoders and CBAM attention for MRI/CT segmentation; Dice "
    "0.91, HD95 3.2 mm, AUROC 0.94 (p<0.001) on BraTS, LiTS and KiTS benchmarks for patients."
)


@pytest.mark.parametrize(
    ("comments", "abstract", "expected"),
    [
        ("This paper has been withdrawn by the authors.", "We study x.", True),
        (None, "This paper has been withdrawn due to an error in Lemma 2.", True),
        ("12 pages, 3 figures", "We study x.", False),
        (float("nan"), None, False),
    ],
)
def test_is_withdrawn(comments, abstract, expected):
    assert is_withdrawn(comments, abstract) is expected


@pytest.mark.parametrize(
    "text",
    [
        "Nous proposons une méthode d'apprentissage profond pour la segmentation des tumeurs "
        "cérébrales chez les patients, évaluée sur des images IRM de trois hôpitaux.",
        "Wir stellen ein neuronales Netz zur Erkennung von Lungenkrebs in Röntgenbildern vor und "
        "vergleichen es mit der Leistung von Radiologen in drei Kliniken.",
        "Proponemos un modelo de aprendizaje automático para predecir la mortalidad hospitalaria "
        "de los pacientes a partir de historias clínicas electrónicas de varios hospitales.",
    ],
)
def test_other_languages_are_flagged(text):
    assert is_non_english(text)


@pytest.mark.parametrize(
    "text",
    [
        ENGLISH,
        TECHNICAL,
        ENGLISH + " Nous proposons un modèle d'apprentissage profond.",
        "Trop court.",
    ],
)
def test_english_bilingual_and_short_texts_are_kept(text):
    assert not is_non_english(text)


def test_near_duplicates_keep_the_smallest_id():
    texts = [
        ENGLISH,
        ENGLISH + " This version corrects a typo.",  # re-post with an addition
        TECHNICAL,
        "A completely different abstract about graph coloring on planar graphs and its bounds.",
        ENGLISH,  # exact copy
    ]
    ids = ["2101.00002", "2101.00001", "2101.00003", "2101.00004", "2101.00005"]
    assert near_duplicates(ids, texts) == {"2101.00002": "2101.00001", "2101.00005": "2101.00001"}


def test_near_duplicates_ignores_empty_and_unique_texts():
    assert near_duplicates(["1", "2", "3"], ["", "one text here", "another text there"]) == {}
    assert near_duplicates([], []) == {}
