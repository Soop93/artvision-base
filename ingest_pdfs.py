"""ingest_pdfs.py construit vector_store_text (RAG du guide) à partir des PDF de pdfs/.

Cycle de vie : script lancé MANUELLEMENT en local, hors de la chaîne d'enrichissement récurrente
(extractAPI -> augmentDB -> vectorisationDB, qui ne gère que les images). Il vectorise les PDF
avec nomic-embed-text dans vector_store_text, qui est ensuite poussé une fois sur S3 ; le Space
le tire au boot comme le reste de la base. On ne le relance que si on ajoute/modifie des PDF.
"""
import os
import glob
import chromadb
import ollama
from pypdf import PdfReader

PDF_DIR = "pdfs"
VECTOR_STORE_PATH = "vector_store_text"
DOCS_COLLECTION_NAME = "art_history_docs"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 800       # caractères par chunk
CHUNK_OVERLAP = 150    # chevauchement entre chunks consécutifs


#Extraction du texte page par page
def extract_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


#Découpage du texte en chunks avec chevauchement. Découpe par caractères bruts : simple et
#suffisant pour ce POC. Piste : passer à langchain-text-splitters (RecursiveCharacterTextSplitter,
#déjà dans les deps) pour couper sur les séparateurs plutôt qu'au milieu des phrases.
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text = " ".join(text.split())  # normalise les espaces/retours à la ligne
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main():
    if not os.path.isdir(PDF_DIR):
        print(f"Dossier introuvable : {PDF_DIR}. Crée-le et places-y tes PDF.")
        return

    #Client Chroma dédié au texte (séparé de vector_store_images)
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_or_create_collection(name=DOCS_COLLECTION_NAME)

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    print(f"{len(pdf_files)} PDF trouvés dans {PDF_DIR}")

    #Contrôle des chunks déjà indexés pour ne pas les refaire
    existing_ids = set(collection.get()["ids"])
    total_indexed = 0

    for pdf_path in pdf_files:
        source_name = os.path.basename(pdf_path)
        print(f"Traitement : {source_name}")

        try:
            pages = extract_pages(pdf_path)
        except Exception as e:
            print(f"  Échec lecture {source_name} : {e}")
            continue

        #Boucle sur chaque page puis chaque chunk de la page
        for page_num, page_text in pages:
            chunks = chunk_text(page_text)
            for c_idx, chunk in enumerate(chunks):
                chunk_id = f"{source_name}_p{page_num}_c{c_idx}"
                if chunk_id in existing_ids:
                    continue

                #Calcul de l'embedding texte via Ollama
                try:
                    resp = ollama.embed(model=EMBED_MODEL, input=chunk)
                    embedding = resp["embeddings"][0]
                except Exception as e:
                    print(f"  Échec embedding {chunk_id} : {e}")
                    continue

                #Insertion du chunk dans ChromaDB
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": source_name,
                        "page": page_num,
                    }],
                )
                existing_ids.add(chunk_id)
                total_indexed += 1

        print(f"  {source_name} : terminé")

    print(f"\nTerminé. {total_indexed} nouveaux chunks indexés.")


if __name__ == "__main__":
    main()
