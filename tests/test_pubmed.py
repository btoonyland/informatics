import xml.etree.ElementTree as ET

import pytest
from helpers import client_for, ok

from informatics.pubmed import (
    earliest_date,
    import_pubmed,
    import_pubmed_files,
    matches_ai,
    parse_article,
    parse_date,
)
from informatics.storage import read_parquet

ESEARCH = b"""<?xml version="1.0" encoding="UTF-8" ?>
<eSearchResult><Count>12345</Count><RetMax>2</RetMax><RetStart>0</RetStart>
<IdList><Id>31000001</Id><Id>31000002</Id></IdList></eSearchResult>"""

EFETCH = """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
<PubmedArticle>
  <MedlineCitation Status="MEDLINE" Owner="NLM">
    <PMID Version="1">31000001</PMID>
    <Article PubModel="Print-Electronic">
      <Journal><Title>Journal of Synthetic Medicine</Title>
        <JournalIssue><PubDate><Year>2020</Year><Month>Mar</Month></PubDate></JournalIssue></Journal>
      <ArticleTitle>Deep learning for <i>tumour</i> detection.</ArticleTitle>
      <ELocationID EIdType="doi" ValidYN="Y">10.1000/jsm.2020.1</ELocationID>
      <Abstract>
        <AbstractText Label="BACKGROUND">Patients need faster diagnosis.</AbstractText>
        <AbstractText Label="METHODS">We trained a neural network.</AbstractText>
      </Abstract>
      <AuthorList>
        <Author><LastName>Kovač</LastName><ForeName>Ana</ForeName></Author>
        <Author><CollectiveName>Synthetic Study Group</CollectiveName></Author>
      </AuthorList>
      <ArticleDate DateType="Electronic">
        <Year>2020</Year><Month>02</Month><Day>14</Day>
      </ArticleDate>
    </Article>
    <MeshHeadingList><MeshHeading>
      <DescriptorName UI="D001185">Artificial Intelligence</DescriptorName>
    </MeshHeading></MeshHeadingList>
  </MedlineCitation>
  <PubmedData><ArticleIdList><ArticleId IdType="pubmed">31000001</ArticleId>
    <ArticleId IdType="doi">10.1000/jsm.2020.1</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle>
<PubmedArticle>
  <MedlineCitation Status="MEDLINE" Owner="NLM">
    <PMID Version="1">31000002</PMID>
    <Article PubModel="Print">
      <Journal><Title>Synthetic Reviews</Title>
        <JournalIssue><PubDate><MedlineDate>2019 Winter</MedlineDate></PubDate></JournalIssue>
      </Journal>
      <ArticleTitle>A review without an abstract.</ArticleTitle>
    </Article>
  </MedlineCitation>
</PubmedArticle>
</PubmedArticleSet>""".encode()


@pytest.mark.parametrize(
    ("xml", "expected"),
    [
        ("<d><Year>2020</Year><Month>Mar</Month></d>", "2020-03"),
        ("<d><Year>2020</Year><Month>02</Month><Day>14</Day></d>", "2020-02"),
        ("<d><Year>2020</Year></d>", "2020"),
        ("<d><MedlineDate>2019 Winter</MedlineDate></d>", "2019"),
        ("<d><MedlineDate>2018 Nov-Dec</MedlineDate></d>", "2018-11"),
        ("<d></d>", None),
    ],
)
def test_parse_date(xml, expected):
    assert parse_date(ET.fromstring(xml)) == expected


def test_parse_article():
    first = parse_article(ET.fromstring(EFETCH).find("PubmedArticle"))
    assert first["id"] == "pmid:31000001"
    assert first["title"] == "Deep learning for tumour detection."
    assert (
        first["abstract"]
        == "Background: Patients need faster diagnosis. Methods: We trained a neural network."
    )
    assert first["authors"] == ["Ana Kovač", "Synthetic Study Group"]
    assert first["categories"] == ["Artificial Intelligence"]
    assert first["created"] == "2020-02" and first["doi"] == "10.1000/jsm.2020.1"
    assert first["journal_ref"] == "Journal of Synthetic Medicine"


def test_import_pubmed_by_year(tmp_path):
    client = client_for([ok(ESEARCH), ok(EFETCH)])
    n = import_pubmed(tmp_path / "pubmed.parquet", [2020], per_year=2, api_key="KEY", client=client)
    assert n == 2
    search = client.session.calls[0]
    assert (search["datetype"], search["mindate"], search["maxdate"]) == ("pdat", "2020", "2020")
    assert search["retmax"] == 2 and search["api_key"] == "KEY" and search["tool"] == "informatics"
    assert client.session.calls[1]["id"] == "31000001,31000002"
    df = read_parquet(tmp_path / "pubmed.parquet").set_index("id")
    # PubMed IDs carry no date, so the year comes from the publication date.
    assert (df.loc["pmid:31000001", "year"], df.loc["pmid:31000001", "month"]) == (2020, 2)
    assert df.loc["pmid:31000002", "year"] == 2019
    assert df.loc["pmid:31000002", "source"] == "pubmed-sample"


def test_earliest_date_prefers_the_first_public_date():
    article = ET.fromstring(
        """<PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
        <Journal><JournalIssue><PubDate><Year>2021</Year><Month>Jan</Month></PubDate></JournalIssue>
        </Journal><ArticleTitle>T</ArticleTitle>
        <ArticleDate DateType="Electronic"><Year>2020</Year><Month>11</Month></ArticleDate>
        </Article></MedlineCitation>
        <PubmedData><History>
          <PubMedPubDate PubStatus="received"><Year>2020</Year><Month>06</Month></PubMedPubDate>
          <PubMedPubDate PubStatus="revised"><Year>2019</Year><Month>01</Month></PubMedPubDate>
          <PubMedPubDate PubStatus="medline"><Year>2018</Year><Month>01</Month></PubMedPubDate>
        </History></PubmedData></PubmedArticle>"""
    )
    # Received in June 2020: earlier than online (Nov 2020) and print (Jan 2021). "revised" and
    # "medline" dates are not publication dates and are ignored.
    assert earliest_date(article) == "2020-06"
    assert parse_article(article)["created"] == "2020-06"


def _article(pmid: int, title: str, mesh: str = "", year: int = 2020) -> str:
    heading = (
        f"<MeshHeadingList><MeshHeading><DescriptorName>{mesh}</DescriptorName></MeshHeading>"
        "</MeshHeadingList>"
        if mesh
        else ""
    )
    return (
        f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><Journal><JournalIssue>"
        f"<PubDate><Year>{year}</Year></PubDate></JournalIssue><Title>J</Title></Journal>"
        f"<ArticleTitle>{title}</ArticleTitle><Abstract><AbstractText>Patients were studied."
        f"</AbstractText></Abstract></Article>{heading}</MedlineCitation></PubmedArticle>"
    )


def _xml_gz(path, body: str) -> None:
    import gzip

    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(f'<?xml version="1.0"?><PubmedArticleSet>{body}</PubmedArticleSet>')


def test_import_pubmed_files_keeps_ai_records_and_applies_updates(tmp_path):
    _xml_gz(
        tmp_path / "pubmed26n0001.xml.gz",
        _article(1, "Deep learning for sepsis")  # keyword
        + _article(2, "A trial of statins", mesh="Machine Learning")  # MeSH heading
        + _article(3, "A trial of statins")  # neither: left out
        + _article(4, "Neural networks for stroke")
        + _article(5, "Random forests for asthma"),  # only a wide-net term
    )
    _xml_gz(
        tmp_path / "pubmed26n0002.xml.gz",  # an update file: newer than the baseline
        _article(1, "Deep learning for sepsis, revised", year=2021)
        + "<DeleteCitation><PMID>4</PMID></DeleteCitation>",
    )
    counts = import_pubmed_files(sorted(tmp_path.glob("*.xml.gz")), tmp_path / "out.parquet")
    assert counts == {"files": 2, "read": 6, "deleted": 1, "kept": 2}
    df = read_parquet(tmp_path / "out.parquet").set_index("id")
    assert sorted(df.index) == ["pmid:1", "pmid:2"]
    assert df.loc["pmid:1", "title"] == "Deep learning for sepsis, revised"
    assert df.loc["pmid:1", "year"] == 2021 and (df["source"] == "pubmed-baseline").all()
    wide = import_pubmed_files(tmp_path.glob("*.xml.gz"), tmp_path / "w.parquet", terms="wide")
    assert wide["kept"] == 3  # adds the random forests paper
    everything = import_pubmed_files(tmp_path.glob("*.xml.gz"), tmp_path / "a.parquet",
                                     everything=True)  # fmt: skip
    assert everything["kept"] == 4  # all but the deleted record


def test_matches_ai():
    record = {"title": "Statins", "abstract": "", "categories": ["Deep Learning"]}
    assert matches_ai(record) and not matches_ai(record, mesh=["Artificial Intelligence"])
    assert matches_ai({"title": "LLMs for triage", "abstract": None, "categories": []})
    assert not matches_ai({"title": "LLMs", "categories": []}, terms="none")
