from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from landedcost import DISCLAIMER
from landedcost.api.app import create_app
from landedcost.config import Settings
from landedcost.container import Services, assemble
from landedcost.embeddings.hashing import HashEmbedder
from landedcost.store.memory import InMemoryStore


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    with TestClient(create_app(services)) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["store"] == "memory"
    assert body["hts_version"] == "hts-2026-09-04"
    assert body["embedder"] == body["index_embedder"] == "hash-v2-c35"
    assert body["llm"] is None


def test_classify(client: TestClient) -> None:
    response = client.post(
        "/classify", json={"description": "Rechargeable lithium-ion battery pack", "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["disclaimer"] == DISCLAIMER
    assert set(body) >= {"candidates", "confidence", "abstained", "abstain_reason", "citations"}
    assert body["abstained"] is False
    assert [c["hts_code"] for c in body["candidates"]] == ["8507.60.00", "8507.60.00.90"]
    assert body["candidates"][0]["citations"] == ["hts:8507.60.00"]
    assert body["citations"][0]["text"].endswith("Lithium-ion batteries")


def test_classify_abstains_on_out_of_scope_text(client: TestClient) -> None:
    body = client.post("/classify", json={"description": "zzqx vvbn plugh"}).json()
    assert body["abstained"] is True
    assert body["candidates"] == []
    assert body["abstain_reason"]


def test_classify_validates_input(client: TestClient) -> None:
    assert client.post("/classify", json={"description": "ab"}).status_code == 422
    assert client.post("/classify", json={"description": "honey", "top_k": 50}).status_code == 422


def test_get_hts_line(client: TestClient) -> None:
    body = client.get("/hts/8507600090").json()
    assert body["hts_code"] == "8507.60.00.90"
    assert body["general_rate"] == "3.4%"
    assert body["rate_inherited_from"] == "8507.60.00"
    assert body["rate_kind"] == "ad_valorem"
    assert body["units"] == ["No.", "kg"]


def test_get_hts_errors(client: TestClient) -> None:
    missing = client.get("/hts/8507.60")
    assert missing.status_code == 404
    assert "8507.60.00" in missing.json()["detail"]
    assert client.get("/hts/12").status_code == 422


def test_landed_cost(client: TestClient) -> None:
    response = client.post(
        "/landed-cost",
        json={
            "hts_code": "8507.60.00.90",
            "customs_value_usd": "12000",
            "mode": "ocean",
            "weight_kg": 850,
            "insurance_usd": 60,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["duty"]["amount_usd"] == "408.00"
    assert body["freight"]["source"] == "static-mock"
    assert body["total_landed_cost_usd"] is not None
    assert any("Chapter 99" in w for w in body["warnings"])


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"hts_code": "8507", "customs_value_usd": 100, "freight_usd": 1}, 422),
        ({"hts_code": "8507.60.00.90", "customs_value_usd": 100}, 422),  # no weight for quote
        ({"hts_code": "8507.60.00.90", "customs_value_usd": -1}, 422),
        ({"hts_code": "8507.60.00.90", "customs_value_usd": "1e30", "freight_usd": 1}, 422),
        ({"hts_code": "8507.60.00.90", "customs_value_usd": 1e100, "freight_usd": 1}, 422),
    ],
)
def test_landed_cost_errors(client: TestClient, payload: dict[str, object], status: int) -> None:
    assert client.post("/landed-cost", json=payload).status_code == status


def test_html_page_and_openapi(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert DISCLAIMER in page.text
    assert "hash-v2-c35" in page.text
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/classify", "/landed-cost", "/hts/{code}", "/healthz"} <= set(paths)


class _MismatchedStore(InMemoryStore):
    async def index_info(self):  # type: ignore[no-untyped-def]
        info = await super().index_info()
        return info.__class__(info.hts_lines, info.rulings, "openai:x:384", 384, info.hts_version)


async def test_embedder_mismatch_is_503(settings: Settings, embedder: HashEmbedder) -> None:
    services = assemble(settings, store=_MismatchedStore(), embedder=embedder)
    with TestClient(create_app(services)) as client:
        response = client.post("/classify", json={"description": "natural honey"})
    assert response.status_code == 503
    assert "re-run" in response.json()["detail"]


def test_app_builds_services_from_settings_in_memory_mode(settings: Settings) -> None:
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/healthz").json()["hts_lines"] > 100
