"""Custom exception handlers for the API."""

from fastapi import status
from fastapi.responses import JSONResponse
from loguru import logger


async def generic_exception_handler(_request, exc: Exception) -> JSONResponse:
    """Handle generic exceptions.

    Args:
        _request: The incoming request (unused, required by FastAPI signature)
        exc: The exception that was raised

    Returns:
        JSONResponse: Error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )


async def validation_exception_handler(_request, exc: Exception) -> JSONResponse:
    """Handle validation exceptions.

    Args:
        _request: The incoming request (unused, required by FastAPI signature)
        exc: The validation exception

    Returns:
        JSONResponse: Error response with validation details
    """
    logger.warning(f"Validation error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
        },
    )
