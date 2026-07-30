import chromadb
import ollama
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

VECTOR_STORE_PATH = "vector_store_text"
DOCS_COLLECTION_NAME = "art_history_docs"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:7b-instruct-q5_K_M")
EMBED_MODEL = "nomic-embed-text"
TOP_K = 4

#Options de génération
GENERATION_OPTIONS = {
    "num_predict": 550,   #Réponses plus longues
    "temperature": 0.4,   
}

SYSTEM_PROMPT = """Tu es un guide de musée passionné et cultivé. Tu t'exprimes à l'oral,
de façon chaleureuse et claire, comme si tu accompagnais un visiteur devant une œuvre.

RÈGLE ABSOLUE : le titre de l'œuvre et le nom de l'artiste, fournis en contexte, sont des faits
vérifiés — ta seule source fiable sur ce qui est réellement représenté.

Tu peux t'appuyer librement sur tes propres connaissances générales d'histoire de l'art
(mouvements, techniques, contexte historique, biographies des artistes...) pour enrichir
tes explications, tant que ça reste cohérent avec le titre et les métadonnées fournis. Les
extraits documentaires fournis en contexte servent à préciser ou compléter tes connaissances,
notamment sur des détails propres à cette collection — utilise-les en priorité quand ils sont pertinents. 
S'il n'y a pas d'extrait pertinent, réponds quand même à partir de tes connaissances générales sur l'artiste et son
mouvement.

Développe ta réponse davantage qu'une simple accroche : vise 8 à 10 phrases par réponse, en 2-3 idées
enchaînées (par exemple contexte historique -> détail technique ou biographique -> ouverture),
plutôt qu'une succession de faits isolés.

Termine TOUJOURS par une seule question courte, simple et concrète : ce que le visiteur
ressent, remarque, ou a envie d'explorer."""


#Connexion ChromaDB PDF, mise en cache : rouvrir un PersistentClient à chaque question est inutile
#et ajoute une latence disque évitable une fois qu'on vise des réponses rapides sur T4.
_DOCS_COLLECTION = None


def _get_docs_collection():
    global _DOCS_COLLECTION
    if _DOCS_COLLECTION is None:
        client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        _DOCS_COLLECTION = client.get_collection(DOCS_COLLECTION_NAME)
    return _DOCS_COLLECTION


#Recherche dans les embeddings des PDFS
def retrieve_context(query: str, n_results: int = TOP_K) -> str:
    try:
        collection = _get_docs_collection()
    except Exception:
        return ""  #Absence d'embeddings

    #Embedding de la question
    resp = ollama.embed(model=EMBED_MODEL, input=query)
    query_embedding = resp["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    #Concatène les passages avec sa source pour le contexte du modèle
    passages = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "source inconnue")
        page = meta.get("page", "?")
        passages.append(f"[{source}, p.{page}]\n{doc}")

    return "\n\n---\n\n".join(passages)


#Instance mise en cache : sur le T4 on veut amortir l'init du client à la 1ère requête puis
#ne plus payer ce coût. keep_alive="60m" garde en plus le modèle chargé en VRAM côté serveur
#Ollama entre deux questions du visiteur (le rechargement d'un 7B est la vraie latence à éviter).
_CHAT_MODEL_INSTANCE = None


def _get_chat_model():
    global _CHAT_MODEL_INSTANCE
    if _CHAT_MODEL_INSTANCE is not None:
        return _CHAT_MODEL_INSTANCE

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        _CHAT_MODEL_INSTANCE = ChatOllama(
            model=CHAT_MODEL, temperature=GENERATION_OPTIONS["temperature"],
            num_predict=GENERATION_OPTIONS["num_predict"],
            num_ctx=8192,          #Contexte élargi (vs 2048 en local) : le T4 a la VRAM pour un 7B
                                    #q5 + 8k de contexte, ce qui laisse la place aux métadonnées +
                                    #5 passages RAG + historique de chat sans troncature silencieuse
            keep_alive="60m")      #Garde le modèle chargé en VRAM 60min entre 2 questions
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        _CHAT_MODEL_INSTANCE = ChatOpenAI(model=CHAT_MODEL, temperature=GENERATION_OPTIONS["temperature"],
                                           max_tokens=GENERATION_OPTIONS["num_predict"])
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        _CHAT_MODEL_INSTANCE = ChatAnthropic(model=CHAT_MODEL, temperature=GENERATION_OPTIONS["temperature"],
                                              max_tokens=GENERATION_OPTIONS["num_predict"])
    else:
        raise ValueError(f"LLM_PROVIDER inconnu : {LLM_PROVIDER}")
    return _CHAT_MODEL_INSTANCE


_ROLE_TO_MESSAGE = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}


#Appel générique au LLM via LangChain, quel que soit le fournisseur configuré
def _call_llm(messages: list) -> str:
    lc_messages = [_ROLE_TO_MESSAGE[m["role"]](content=m["content"]) for m in messages]
    response = _get_chat_model().invoke(lc_messages)
    return response.content


class GuideState(TypedDict):
    artwork_metadata: dict
    question: str
    chat_history: list
    context: str
    reply: str


#La requête d'embedding envoyée à Chroma est enrichie du contexte de l'œuvre (titre/artiste/
#période), même pour les questions de suivi. 
def _node_retrieve(state: GuideState) -> GuideState:
    artwork_metadata = state["artwork_metadata"]
    artwork_context = f"{artwork_metadata.get('title', '')} {artwork_metadata.get('artist', '')} " \
                       f"{artwork_metadata.get('period', '') or ''}".strip()
    if state["question"]:
        query = f"{artwork_context} — {state['question']}" if artwork_context else state["question"]
    else:
        query = artwork_context
    state["context"] = retrieve_context(query)
    return state


def _node_generate(state: GuideState) -> GuideState:
    artwork_metadata = state["artwork_metadata"]
    context = state["context"]

#Toutes les métadonnées disponibles, mises à plat en texte pour le modèle.
    metadata_fields = {
        "Titre": artwork_metadata.get("title"),
        "Artiste": artwork_metadata.get("artist"),
        "Date": artwork_metadata.get("dated"),
        "Médium": artwork_metadata.get("medium"),
        "Technique": artwork_metadata.get("technique"),
        "Classification": artwork_metadata.get("classification"),
        "Période": artwork_metadata.get("period"),
        "Culture": artwork_metadata.get("culture"),
        "Dimensions": artwork_metadata.get("dimensions"),
        "Département": artwork_metadata.get("department"),
        "Provenance": artwork_metadata.get("creditline"),
        "Description (base)": artwork_metadata.get("description"),
        "Texte du cartel": artwork_metadata.get("labeltext"),
    }
    metadata_text = "\n".join(f"- {k} : {v}" for k, v in metadata_fields.items() if v)

    if state["question"]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({
            "role": "system",
            "content": f"Métadonnées vérifiées de l'œuvre actuellement présentée :\n{metadata_text}\n\n"
                        f"Extraits documentaires :\n{context}",
        })
        messages.extend(state["chat_history"])
        messages.append({"role": "user", "content": state["question"]})
        messages.append({
            "role": "system",
            "content": "Développe ta réponse en 8-10 phrases (contexte, détail précis, ouverture), "
                       "puis termine par UNE seule question simple et concrète sur le ressenti ou "
                       "l'observation du visiteur (jamais une question abstraite sur l'importance "
                       "ou la portée d'un concept).",
        })
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"Métadonnées vérifiées de l'œuvre présentée :\n{metadata_text}",
            },
            {
                "role": "user",
                "content": (
                    f"Extraits documentaires disponibles :\n{context}\n\n"
                    f"Présente cette œuvre au visiteur comme le ferait un guide, "
                    f"de façon vivante et engageante, en 8-10 phrases développées, "
                    f"et termine par une question ouverte pour l'inviter à réagir."
                ),
            },
        ]

    state["reply"] = _call_llm(messages)
    return state


_graph = StateGraph(GuideState)
_graph.add_node("retrieve", _node_retrieve)
_graph.add_node("generate", _node_generate)
_graph.set_entry_point("retrieve")
_graph.add_edge("retrieve", "generate")
_graph.add_edge("generate", END)
_guide_graph = _graph.compile()


#Génèration de l'introduction de l'œuvre identifiée
def generate_intro(artwork_metadata: dict) -> str:
    result = _guide_graph.invoke({
        "artwork_metadata": artwork_metadata,
        "question": "",
        "chat_history": [],
        "context": "",
        "reply": "",
    })
    return result["reply"]


#Répond à une question du visiteur, en s'appuyant sur le RAG + l'historique de la conversation
def answer_question(question: str, artwork_metadata: dict, chat_history: list) -> str:
    result = _guide_graph.invoke({
        "artwork_metadata": artwork_metadata,
        "question": question,
        "chat_history": chat_history,
        "context": "",
        "reply": "",
    })
    return result["reply"]