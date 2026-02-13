import rdflib
from rdflib.namespace import RDF, RDFS, OWL

from dataclasses import dataclass

@dataclass
class FormalConcept:
    extent: set[str]
    intent: set[str]
    min_generators: set[str]
    label: str = None
    uri_ref: rdflib.URIRef = None



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
        namespace: rdflib.Namespace=None, concept_ref: rdflib.URIRef = None, label: str = None
) -> tuple[rdflib.URIRef, list[tuple[rdflib.Node, rdflib.Node, rdflib.Node]]]:
    if label is None:
        label = '_'.join(sorted({attr_ref.fragment for min_generator in min_generators for attr_ref in min_generator}))
        if not label:
            label = 'TopConcept'

    concept_ref = rdflib.URIRef(namespace[label]) if concept_ref is None else concept_ref

    triples = []
    triples.append((concept_ref, RDF.type, OWL.Class))
    triples.append((concept_ref, RDFS.label, rdflib.Literal(label)))
    for min_generator in min_generators:
        if not min_generator: continue
        mingen_ref, mingen_triples = description_to_triples(min_generator)
        triples.append((concept_ref, OWL.equivalentClass, mingen_ref))
        for triple in mingen_triples: triples.append(triple)

    for attr_ref in intent:
        triples.append((concept_ref, RDFS.subClassOf, attr_ref))

    for obj_ref in extent:
        triples.append((obj_ref, RDF.type, concept_ref))

    return concept_ref, triples


def construct_ontology(
        namespace: rdflib.Namespace, prefix: str,
        objects_to_refs: dict[str, rdflib.URIRef],
        attributes_to_refs: dict[str, rdflib.URIRef],
        incidence_relation: set[tuple[str, str]],
        concepts: list[FormalConcept]
) -> rdflib.Graph:
    obj_refs = {obj_ref: obj for obj, obj_ref in objects_to_refs.items()}
    attr_refs = {attr_ref: attr for attr, attr_ref in attributes_to_refs.items()}
    incidence_refs = {(objects_to_refs[obj], attributes_to_refs[attr]) for obj, attr in incidence_relation}

    kgraph = rdflib.Graph()
    kgraph.bind(prefix, namespace)

    kgraph.add( (rdflib.URIRef(namespace), RDF.type, OWL.Ontology) )

    for triple in context_to_triples(obj_refs, attr_refs, incidence_refs):
        kgraph.add(triple)

    for concept in concepts:
        extent_refs = {objects_to_refs[obj] for obj in concept.extent}
        intent_refs = {attributes_to_refs[attr] for attr in concept.intent}
        min_gens_refs = [{attributes_to_refs[attr] for attr in mingen} for mingen in concept.min_generators]
        concept_ref, concept_triples = concept_to_triples(
            extent_refs, intent_refs, min_gens_refs, namespace, concept.uri_ref, concept.label
        )
        concept.uri_ref = concept_ref
        for triple in concept_triples:
            kgraph.add(triple)
    return kgraph


