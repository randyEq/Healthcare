Place your PDF files here.

## How to use

1. Upload/copy your PDF files into this folder:
   ```
   /workspace/ingestion/documents/
   ```

2. Run the ingestion pipeline to build the FAISS index:
   ```bash
   cd /workspace
   python3 -m ingestion.ingest --dir /workspace/ingestion/documents
   ```

3. Or ingest everything (sample data + PDFs):
   ```bash
   python3 -m ingestion.ingest
   ```

4. Verify the index:
   ```bash
   python3 -m ingestion.ingest --verify
   ```

The FAISS index will be saved to /workspace/vector_db/ and the RAG agent
will automatically pick it up on the next query.

Supported file formats: PDF (.pdf), Text (.txt), JSON (.json)
