import fastapi
import io
import PIL

from models import get_top5


router = fastapi.APIRouter()

@router.post("/search")
async def search(request: fastapi.Request, file: fastapi.UploadFile = fastapi.File(...)):
    contents = await file.read()
    try:
        image = PIL.Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return fastapi.responses.JSONResponse(status_code=400, content={"error": "Invalid image format"})
    results = get_top5(image, request.app.state)
    return {"results": results}
