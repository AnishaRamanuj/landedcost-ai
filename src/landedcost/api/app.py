"""HTTP API: ``POST /classify``, ``POST /landed-cost``, ``GET /hts/{code}``, ``GET /healthz``.

``create_app`` accepts pre-built :class:`Services` so tests can inject an in-memory store
and fake providers; otherwise services are built from the environment at startup.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import InvalidOperation
from importlib import resources
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader

from landedcost import DISCLAIMER, __version__
from landedcost.api.schemas import ClassifyRequest, HealthOut, HtsLineOut
from landedcost.classify.models import ClassificationResult
from landedcost.config import Settings
from landedcost.container import Services, build_services
from landedcost.hts.codes import HtsCode, InvalidHtsCodeError
from landedcost.landed.calculator import LandedCostError
from landedcost.landed.freight import FreightQuoteError
from landedcost.landed.models import LandedCostRequest, LandedCostResult
from landedcost.rates.parser import parse_rate
from landedcost.store.base import EmbedderMismatchError, StoreUnavailableError

_TEMPLATES = Jinja2Templates(
    env=Environment(
        loader=FileSystemLoader(str(resources.files("landedcost.api") / "templates")),
        autoescape=True,
    )
)


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.services = services
            yield
            return
        built = await build_services(settings or Settings())
        app.state.services = built
        try:
            yield
        finally:
            await built.aclose()

    app = FastAPI(
        title="landedcost-ai",
        version=__version__,
        description=(
            "HTS classification suggestions with cited tariff text and CBP rulings, an "
            f"abstention gate, and landed-cost estimates. {DISCLAIMER}"
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(EmbedderMismatchError)
    async def _embedder_mismatch(_: Request, exc: EmbedderMismatchError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(StoreUnavailableError)
    async def _store_unavailable(_: Request, exc: StoreUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request, svc: ServicesDep) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "version": __version__,
                "disclaimer": DISCLAIMER,
                "embedder": svc.embedder.name,
                "llm": svc.llm.label if svc.llm else "none (extractive answers)",
            },
        )

    @app.get("/healthz", response_model=HealthOut)
    async def healthz(svc: ServicesDep) -> HealthOut:
        info = await svc.store.index_info()
        return HealthOut(
            status="ok" if info.hts_lines else "empty_index",
            store=svc.settings.store_backend.value,
            hts_lines=info.hts_lines,
            rulings=info.rulings,
            hts_version=info.hts_version,
            embedder=svc.embedder.name,
            index_embedder=info.embedder,
            llm=svc.llm.label if svc.llm else None,
            retrieval_mode=svc.settings.retrieval_mode.value,
        )

    @app.post("/classify", response_model=ClassificationResult)
    async def classify(body: ClassifyRequest, svc: ServicesDep) -> ClassificationResult:
        return await svc.classifier.classify(body.description, top_k=body.top_k)

    @app.post("/landed-cost", response_model=LandedCostResult)
    async def landed_cost(body: LandedCostRequest, svc: ServicesDep) -> LandedCostResult:
        try:
            return await svc.landed_cost.estimate(body)
        except LandedCostError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FreightQuoteError as exc:
            raise HTTPException(status_code=502 if exc.upstream else 422, detail=str(exc)) from exc
        except InvalidOperation as exc:  # defence in depth: request bounds should prevent it
            raise HTTPException(status_code=422, detail="amount out of range") from exc

    @app.get("/hts/{code}", response_model=HtsLineOut)
    async def get_hts(code: str, svc: ServicesDep) -> HtsLineOut:
        try:
            canonical = HtsCode.parse(code).dotted
        except InvalidHtsCodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        line = (await svc.store.get_hts_lines([canonical])).get(canonical)
        if line is None:
            nearby = await svc.store.find_hts_by_prefix(canonical, 10)
            detail = f"HTS {canonical} not found"
            if nearby:
                detail += f"; lines under it: {', '.join(n.code for n in nearby)}"
            raise HTTPException(status_code=404, detail=detail)
        return HtsLineOut(
            hts_code=line.code,
            level=line.level.value,
            description=line.description,
            breadcrumb=line.breadcrumb,
            units=list(line.units),
            general_rate=line.general_rate,
            special_rate=line.special_rate,
            other_rate=line.other_rate,
            rate_inherited_from=line.rate_source_code if line.rate_inherited else None,
            rate_kind=parse_rate(line.general_rate).kind.value,
            footnotes=list(line.footnotes),
        )

    return app
