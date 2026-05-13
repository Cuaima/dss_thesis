import re
import spacy
from typing import List, Dict, Tuple

nlp = spacy.load("nl_core_news_lg")

def recognize_entities(text: str) -> List[Dict]:
    """
    Identifies various types of entities in a given text using NER and regex for URLs.

    Args:
        text: The text to be processed.

    Returns:
        A list of dictionaries, where each dictionary represents an entity and contains the following keys:
        - type: The type of entity (e.g., "PERSON", "ORGANIZATION", "LOCATION", "PHONE", "EMAIL", "URL", "PRODUCT").
        - start: The starting index of the entity in the text.
        - end: The ending index of the entity in the text.
        - text: The original text of the entity.
    """
    doc = nlp(text)
    entities = []

    # NER-based entity recognition
    for ent in doc.ents:
        if ent.label_ not in ["MONEY"]:
            entity = {
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "text": ent.text,
            }
            entities.append(entity)

    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    for pattern, entity_type in [
        (url_pattern, "URL"),
        (email_pattern, "EMAIL"),
    ]:
        for match in pattern.finditer(text):
            entity = {
                "type": entity_type,
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
            }
            entities.append(entity)

    # Sort entities by their starting position
    entities.sort(key=lambda x: x["start"])

    return entities

def anonymize(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Anonymizes the given text by replacing identified entities with placeholders.

    Args:
        text: The text to be anonymized.

    Returns:
        A tuple containing the anonymized text and the anonymization map.
    """
    entities = recognize_entities(text)
    anonymization_map = {}
    entity_counters = {}
    for entity in reversed(entities):  # Process entities from end to start
        entity_type = entity['type']
        if entity_type not in entity_counters:
            entity_counters[entity_type] = 1
        else:
            entity_counters[entity_type] += 1
        placeholder = f"[ENTITY_{entity_type}_{entity_counters[entity_type]}]"
        anonymization_map[placeholder] = entity["text"]
        text = text[: entity["start"]] + placeholder + text[entity["end"] :]
    return text, anonymization_map

def deanonymize(anonymized_text: str, anonymization_map: Dict[str, str]) -> str:
    """
    Restores the original text by replacing placeholders with their corresponding original entities.

    Args:
        anonymized_text: The anonymized text.
        anonymization_map: A dictionary that maps placeholders to their corresponding original entities.

    Returns:
        The de-anonymized text.
    """
    for placeholder, original_text in anonymization_map.items():
        anonymized_text = anonymized_text.replace(placeholder, original_text)
    return anonymized_text
