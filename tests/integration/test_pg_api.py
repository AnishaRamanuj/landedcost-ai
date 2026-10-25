import pytest
from fastapi.testclient import TestClient

from landedcost.api.app import create_app
from landedcost.config import Settings, StoreBackend
from landedcost.store.postgres import PostgresStore

pytestmark = pytest.mark.integration


def test_api_end_to_end_on_postgres(pg_store: PostgresStore, pg_schema: str) -> None:
    settings = Settings(
        _env_file=None,
        store_backend=StoreBackend.POSTGRES,
        database_url=pg_schema,
    )
    with TestClient(create_app(settings=settings)) as client:
        health = client.get("/healthz").json()
        assert health["store"] == "postgres"
        assert health["hts_lines"] > 100
        assert health["index_embedder"] == "hash-v2-c35"

        classified = client.post(
            "/classify", json={"description": "Rechargeable lithium-ion battery pack"}
        ).json()
        assert classified["abstained"] is False
        assert classified["candidates"][0]["hts_code"].startswith("8507.60")

        line = client.get("/hts/8507.60.00.90").json()
        assert line["rate_inherited_from"] == "8507.60.00"

        cost = client.post(
            "/landed-cost",
            json={"hts_code": "8507.60.00.90", "customs_value_usd": 12000, "freight_usd": 850},
        ).json()
        assert cost["total_landed_cost_usd"] == "13314.57"
