from fastapi import FastAPI, UploadFile, File, HTTPException, Body
import uvicorn
from services.pdf_processor import process_pdf
from services.vector_store import save_chunks_to_db
from services.qa_engine import ask_question

app = FastAPI(title='Rag Intelligence API', description='A backend API for pdf ingestion and AI querying')

@app.post('/api/v1/upload')
async def upload_document(file: UploadFile = File()):
    """Endpoint to upload a pdf, chunk it and store embeddings in postgreSQL."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only pdfs are allowed')
    
    try:
        file_bytes = await file.read()

        chunks = process_pdf(file_bytes=file_bytes, filename=file.filename)

        save_chunks_to_db(chunks=chunks)

        return {
            'message': 'Document processed and saved successfully to vector database',
            'filename': file.filename,
            'total_chunks': len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to process pdf: {str(e)}')
    
@app.post('/api/v1/query')
async def query_document(question: str = Body(..., embed=True)):
    """Endpoint to ask questiona"""
    try:
        response = ask_question(question=question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Query failed: {str(e)}')


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)