from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
def health() -> dict[str, str]:
    return {"status": "ok"}
