import xml.etree.ElementTree as ET

import pytest
from helpers import client_for, ok

from informatics.pubmed import import_pubmed, parse_article, parse_date
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
    assert df.loc["pmid:31000002", "year"] == 2019 and df.loc["pmid:31000002", "source"] == "pubmed"
