# !pip install networkx matplotlib torch transformers sentence-transformers lmdb -q
# !pip install tf_keras

import networkx as nx
import numpy as np
import pickle
from pathlib import Path
from tqdm.auto import tqdm

import torch
import lmdb
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

import rdflib

def get_label(t: rdflib.term, kg: rdflib.Graph) -> str:
    if isinstance(kg, rdflib.Literal):
        return t.value

    t_label = list(kg.triples((t, rdflib.RDFS.label, None)))
    if not t_label:
        return t.title()

    return str(t_label[0][2])


def compute_embeddings(
        graph_path: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        lmdb_path: str = './embeddings_local'
) -> SentenceTransformer:
    # Create directory if needed
    graph = rdflib.Graph().parse(graph_path)
    entities = {node: get_label(node, graph) for node in graph.all_nodes() if not isinstance(node, rdflib.BNode)}

    Path(lmdb_path).mkdir(parents=True, exist_ok=True)

    embedding_model = SentenceTransformer(embedding_model)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    embeddings = {qid: embedding_model.encode(label) for qid, label in tqdm(entities.items(), desc='Compute embeddings')}

    # compute database size
    map_size = len(embeddings) * embedding_dim * 8 * 10
    # Open LMDB database
    env = lmdb.open(lmdb_path, map_size=map_size, max_dbs=0)
    # Write embeddings
    with env.begin(write=True) as txn:
        for qid, embedding in embeddings.items():
            # Serialize the embedding
            embedding_bytes = pickle.dumps(embedding)
            # Store in LMDB
            txn.put(qid.encode("utf-8"), embedding_bytes)

    env.close()
    print(f"LMDB database created: {lmdb_path}")
    return embedding_model


class GraphRAG:
    def __init__(self, triples_file, lmdb_path, embedding_model):
        self.lmdb_path = lmdb_path
        self.embedding_model = embedding_model
        self.graph = nx.DiGraph()

        # Open LMDB database
        self.env = lmdb.open(lmdb_path, readonly=True)

        # Load graph
        self.load_graph(triples_file)

        print(f"Graph loaded!")
        print(f"{self.graph.number_of_nodes()} nodes")
        print(f"{self.graph.number_of_edges()} edges")

    def load_graph(self, filepath):
        kgraph = rdflib.Graph()
        kgraph.parse(filepath)
        for s, p, o in kgraph:
            if s not in self.graph: self.graph.add_node(s, label=get_label(s, kgraph))
            if o not in self.graph: self.graph.add_node(o, label=get_label(o, kgraph))

            self.graph.add_edge(s, o, relation=p, relation_label=get_label(p, kgraph))

    def get_embedding(self, qid: str) -> np.ndarray:
        with self.env.begin() as txn:
            value = txn.get(qid.encode("utf-8"))

            if value:
                embedding = pickle.loads(value)
                return embedding

            return None

    def encode_query(self, query: str) -> np.ndarray:
        return self.embedding_model.encode(query)

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if emb1 is None or emb2 is None:
            return 0.0

        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def semantic_search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        # Encode the query
        query_embedding = self.encode_query(query)

        # To compute the similarity with all entities
        similarities = []

        for node in self.graph.nodes():
            # Retrieve embedding from LMDB
            node_embedding = self.get_embedding(node)

            if node_embedding is not None:
                # Compute similarity
                score = self.cosine_similarity(query_embedding, node_embedding)
                similarities.append((node, score))
        # Sort by descending score
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Return top-k
        return similarities[:top_k]

    def expand_subgraph(self, seed_nodes: list[str], hops: int = 2) -> nx.DiGraph:
        nodes = set(seed_nodes)
        for _ in range(hops):
            new_nodes = set()

            for node in nodes:
                new_nodes.update(self.graph.successors(node))
                new_nodes.update(self.graph.predecessors(node))
            nodes.update(new_nodes)

        return self.graph.subgraph(nodes).copy()

    def subgraph_to_facts(self, subgraph: nx.DiGraph) -> list[str]:
        facts = []
        for u, v, data in subgraph.edges(data=True):
            u_label = subgraph.nodes[u].get("label", u)
            v_label = subgraph.nodes[v].get("label", v)
            rel_label = data.get("relation_label", "related to")

            fact = f"{u_label} {rel_label} {v_label}"
            facts.append(fact)

        return facts

    def query(self, question: str, top_k: int = 5, hops: int = 2) -> dict:
        print(f"Query: {question}")

        # Semantic search (w/ embeddings)
        print("Semantic search for entities ...")
        results = self.semantic_search(question, top_k=top_k)

        if not results:
            return {
                "question": question,
                "entities": [],
                "scores": [],
                "subgraph": None,
                "facts": [],
            }

        seed_nodes = [qid for qid, score in results]
        scores = [score for qid, score in results]

        print(f"{len(seed_nodes)} entities found")
        for qid, score in results[:3]:
            label = self.graph.nodes[qid].get("label", qid)
            print(f"{label} -> similarity score: {score})")

        # Expansion
        print(f"Graph expansion {hops}-hop ...")
        subgraph = self.expand_subgraph(seed_nodes, hops=hops)

        print(f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

        # Convert to facts
        facts = self.subgraph_to_facts(subgraph)

        return {
            "question": question,
            "entities": seed_nodes,
            "scores": scores,
            "subgraph": subgraph,
            "facts": facts[:20],  # You can change this value to increase the context
        }


def initialise_llm(model_name="Qwen/Qwen2-1.5B-Instruct") -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    model = model.to(device)

    print("Model loaded!")
    return tokenizer, model


def generate_answer(question: str, facts: list[str], tokenizer: AutoTokenizer, model: AutoModelForCausalLM, max_tokens: int = 200):
    context = "\n".join([f"- {fact}" for fact in facts[:15]])
    if not context:
        return "Sorry, I couldn't find relevant information."

    messages = [
        {
            "role": "system",
            "content": "You answer questions in English based ONLY on the provided facts from the knowledge graph. Be concise and factual."
        },
        {
            "role": "user",
            "content": f"""Facts from knowledge graph:

            {context}

            Question: {question}

            Answer using ONLY the facts above."""
        }
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,

            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in response:
        answer = response.split("assistant")[-1].strip()
    else:
        answer = response.strip()

    return answer


def graphrag_query(graphrag: GraphRAG, tokenizer, model, question: str, top_k: int = 5, hops: int = 2):
    # Retrieval + Expansion
    result = graphrag.query(question, top_k=top_k, hops=hops)

    if not result["facts"]:
        print("Answer: No relevant information found.")
        return result

    # Generation
    print("Generating answer...")
    answer = generate_answer(question, result["facts"], tokenizer, model)
    result["answer"] = answer

    print("Answer:")
    print(answer)

    return result

# Initialize
# graphrag = GraphRAG('film-ontology-full.ttl', LMDB_PATH, embedding_model)