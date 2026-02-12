import rdflib

from rdflib.namespace import RDF, RDFS, OWL


from dataclasses import dataclass

@dataclass
class FormalConcept:
    extent: set[rdflib.URIRef]
    intent: set[rdflib.URIRef]
    min_generators: set[frozenset[rdflib.URIRef]]

    def to_triples(self) -> list[tuple[rdflib.Node, rdflib.Node, rdflib.Node]]:
        pass



def context_to_triples(
        objects_refs: dict[rdflib.URIRef, str], attributes_refs: dict[rdflib.URIRef, str],
        incidence: set[tuple[rdflib.URIRef, rdflib.URIRef]],
) -> list[tuple[rdflib.Node, rdflib.Node, rdflib.Node]]:

    triples = []
    for m_ref, m_label in attributes_refs.items():
        triples.append((m_ref, RDFS.subClassOf, OWL.Class))
        triples.append((m_ref, RDFS.label, rdflib.Literal(m_label)))

    for g_ref, g_label in objects_refs.items():
        triples.append((g_ref, RDF.type, OWL.NamedIndividual))
        triples.append((g_ref, RDFS.label, rdflib.Literal(g_label)))

    for (g_ref, m_ref) in incidence:
        triples.append((g_ref, RDF.type, m_ref))
    return triples


def description_to_triples(
        attribute_refs: set[rdflib.URIRef]) -> tuple[rdflib.BNode, list[tuple[rdflib.Node, rdflib.Node, rdflib.Node]]]:
    assert attribute_refs
    description = sorted(attribute_refs)
    intersection_node = rdflib.BNode()
    conj_nodes = [rdflib.BNode() for _ in description]

    triples = []
    triples.append((intersection_node, RDF.type, OWL.Class))
    triples.append((intersection_node, OWL.intersectionOf, conj_nodes[0]))
    for i, (m_ref, conj_node) in enumerate(zip(description, conj_nodes)):
        triples.append((conj_node, RDF.first, m_ref))
    for conj_node, conj_rest in zip(conj_nodes, conj_nodes[1:]):
        triples.append((conj_node, RDF.rest, conj_rest))
    triples.append((conj_nodes[-1], RDF.rest, RDF.nil))
    return intersection_node, triples


def concept_to_triples(
        extent: set[rdflib.URIRef], intent: set[rdflib.URIRef], min_generators: list[set[rdflib.URIRef]],
        prefix: str
) -> tuple[rdflib.URIRef, list[tuple[rdflib.Node, rdflib.Node, rdflib.Node]]]:
    assert prefix.endswith(':')


    label = '_'.join(sorted({attr_ref.title() for min_generator in min_generators for attr_ref in min_generator}))
    concept_ref = rdflib.URIRef(f"{prefix}{label}")

    triples = []
    triples.append((concept_ref, RDF.type, OWL.Class))
    triples.append((concept_ref, RDFS.label, rdflib.Literal(label)))
    for min_generator in min_generators:
        mingen_ref, mingen_triples = description_to_triples(min_generator)
        triples.append((concept_ref, OWL.equivalentClass, mingen_ref))
        for triple in mingen_triples: triples.append(triple)

    for attr_ref in intent:
        triples.append((concept_ref, RDFS.subClassOf, attr_ref))

    return concept_ref, triples



