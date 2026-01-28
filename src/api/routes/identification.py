"""Camera identification endpoints."""

import time
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, Query, HTTPException
from loguru import logger

from src.api.dependencies import get_identification_service
from src.api.models.responses import IdentifyResponse
from src.identification.service import IdentificationService

router = APIRouter()


@router.post("/identify", response_model=IdentifyResponse)
async def identify_camera(
    image: UploadFile = File(..., description="Image file to analyze"),
    top_k: int = Query(
        default=5, ge=1, le=20, description="Number of top matches to return"
    ),
    min_similarity: float = Query(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0-1.0)",
    ),
    service: IdentificationService = Depends(get_identification_service),
) -> IdentifyResponse:
    r"""Identify cameras in an uploaded image.

    This endpoint accepts an image file, detects cameras using YOLOv8,
    and retrieves the most similar cameras from the catalog using CLIP embeddings.

    :param image: The image file to analyze (JPEG, PNG)
    :param top_k: Maximum number of matches to return per detection
    :param min_similarity: Minimum cosine similarity score for matches
    :param service: Injected identification service
    :return: Detection results with matched cameras
    :raises HTTPException: If image processing fails

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/identify?top_k=5&min_similarity=0.3" \\
          -F "image=@path/to/photo.jpg"
        ```
    """
    start_time = time.time()

    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Expected image file.",
        )

    tmp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await image.read()
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        logger.info(f"Processing image: {image.filename} (size: {len(contents)} bytes)")

        # Run identification with min_similarity parameter
        results = service.identify_from_image(
            tmp_path, top_k=top_k, min_similarity=min_similarity
        )

        processing_time = (time.time() - start_time) * 1000

        logger.success(
            f"Identified {len(results)} detections in {processing_time:.2f}ms"
        )

        return IdentifyResponse(
            success=True,
            detections_count=len(results),
            results=results,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    finally:
        # Cleanup temporary file
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
