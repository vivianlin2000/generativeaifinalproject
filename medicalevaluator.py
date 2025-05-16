"""
Evaluation module for assessing medical conversation quality.

This module implements metrics and evaluation methods specific to
multi-turn medical conversations, focused on context retention,
consistency, and appropriate escalation recognition.
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import torch
from transformers import PreTrainedModel, AutoTokenizer
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import json
import os

from model.model_utils import (
    context_retention_test,
    evaluate_medical_consistency,
    check_symptom_escalation
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MedicalConversationEvaluator:
    """
    Evaluator for medical conversation quality.

    This class implements:
    1. Context retention metrics
    2. Consistency evaluations
    3. Escalation appropriateness checks
    4. Medical information accuracy assessment
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        output_dir: str = "./evaluation_results"
    ):
        """
        Initialize the evaluator.

        Args:
            model: The model to evaluate
            tokenizer: The model's tokenizer
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Reference symptoms by body system for testing
        self.system_symptoms = {
            "respiratory": ["cough", "shortness of breath", "wheezing", "sore throat"],
            "digestive": ["nausea", "vomiting", "diarrhea", "abdominal pain"],
            "neurological": ["headache", "dizziness", "numbness", "confusion"],
            "musculoskeletal": ["joint pain", "muscle weakness", "back pain", "stiffness"],
            "cardiovascular": ["chest pain", "palpitations", "edema", "high blood pressure"]
        }

        # Reference conditions for consistency testing
        self.test_conditions = [
            {"condition": "common cold", "symptoms": ["runny nose", "cough", "sore throat"]},
            {"condition": "migraine", "symptoms": ["headache", "nausea", "light sensitivity"]},
            {"condition": "bronchitis", "symptoms": ["cough", "chest discomfort", "fatigue"]},
            {"condition": "gastroenteritis", "symptoms": ["nausea", "vomiting", "diarrhea", "abdominal cramps"]}
        ]

        # Medical emergency scenarios for escalation testing
        self.escalation_scenarios = [
            {
                "scenario": "chest_pain",
                "initial_symptoms": ["mild chest discomfort", "fatigue"],
                "severe_symptoms": ["severe chest pain radiating to left arm", "shortness of breath", "sweating"]
            },
            {
                "scenario": "head_injury",
                "initial_symptoms": ["headache", "small bump on head"],
                "severe_symptoms": ["confusion", "repeated vomiting", "unequal pupils"]
            },
            {
                "scenario": "infection",
                "initial_symptoms": ["low-grade fever", "mild sore throat"],
                "severe_symptoms": ["fever of 104°F", "severe neck stiffness", "confusion"]
            },
            {
                "scenario": "abdominal_pain",
                "initial_symptoms": ["mild stomach discomfort", "bloating"],
                "severe_symptoms": ["severe right lower abdominal pain", "fever", "vomiting"]
            }
        ]

    def evaluate_context_retention(
        self,
        n_examples: int = 5,
        context_markers: bool = True,
        save_results: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the model's context retention capabilities.

        Args:
            n_examples: Number of examples to test
            context_markers: Whether to use context markers in prompts
            save_results: Whether to save detailed results

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating context retention with {n_examples} examples (context_markers={context_markers})")

        results = []
        total_score = 0.0

        # Test across different body systems
        systems = list(self.system_symptoms.keys())
        for i in range(n_examples):
            system = systems[i % len(systems)]
            symptoms = self.system_symptoms[system]

            # Use two symptoms from the system
            test_symptoms = symptoms[:2]

            # Run the context retention test
            conversation_log, retention_score = context_retention_test(
                self.model,
                self.tokenizer,
                test_symptoms,
                context_markers=context_markers
            )

            results.append({
                "example_id": i,
                "system": system,
                "symptoms": test_symptoms,
                "context_markers": context_markers,
                "retention_score": retention_score,
                "conversation_log": conversation_log
            })

            total_score += retention_score

        # Calculate average score
        avg_score = total_score / n_examples

        # Save detailed results if requested
        if save_results:
            output_path = os.path.join(
                self.output_dir,
                f"context_retention_results_markers={context_markers}.json"
            )
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Saved detailed context retention results to {output_path}")

        return {
            "context_retention_score": avg_score,
            "n_examples": n_examples,
            "context_markers": context_markers
        }

    def evaluate_consistency(
        self,
        n_examples: int = 4,
        context_markers: bool = True,
        save_results: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the model's consistency when asked similar questions.

        Args:
            n_examples: Number of examples to test
            context_markers: Whether to use context markers
            save_results: Whether to save detailed results

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating consistency with {n_examples} examples (context_markers={context_markers})")

        results = []
        total_score = 0.0

        # Test with different conditions
        for i in range(min(n_examples, len(self.test_conditions))):
            condition = self.test_conditions[i]["condition"]
            symptoms = self.test_conditions[i]["symptoms"]

            # Run the consistency test
            test_log, consistency_score = evaluate_medical_consistency(
                self.model,
                self.tokenizer,
                condition,
                symptoms,
                context_markers=context_markers
            )

            results.append({
                "example_id": i,
                "condition": condition,
                "symptoms": symptoms,
                "context_markers": context_markers,
                "consistency_score": consistency_score,
                "test_log": test_log
            })

            total_score += consistency_score

        # Calculate average score
        avg_score = total_score / n_examples

        # Save detailed results if requested
        if save_results:
            output_path = os.path.join(
                self.output_dir,
                f"consistency_results_markers={context_markers}.json"
            )
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Saved detailed consistency results to {output_path}")

        return {
            "consistency_score": avg_score,
            "n_examples": n_examples,
            "context_markers": context_markers
        }

    def evaluate_escalation(
        self,
        n_examples: int = 4,
        context_markers: bool = True,
        save_results: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate whether the model appropriately escalates recommendations
        when symptoms worsen.

        Args:
            n_examples: Number of examples to test
            context_markers: Whether to use context markers
            save_results: Whether to save detailed results

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating escalation recognition with {n_examples} examples (context_markers={context_markers})")

        results = []
        correct_escalations = 0

        # Test with different escalation scenarios
        for i in range(min(n_examples, len(self.escalation_scenarios))):
            scenario = self.escalation_scenarios[i]["scenario"]
            initial_symptoms = self.escalation_scenarios[i]["initial_symptoms"]
            severe_symptoms = self.escalation_scenarios[i]["severe_symptoms"]

            # Run the escalation test
            test_log, escalation_detected = check_symptom_escalation(
                self.model,
                self.tokenizer,
                initial_symptoms,
                severe_symptoms,
                context_markers=context_markers
            )

            results.append({
                "example_id": i,
                "scenario": scenario,
                "initial_symptoms": initial_symptoms,
                "severe_symptoms": severe_symptoms,
                "context_markers": context_markers,
                "escalation_detected": escalation_detected,
                "test_log": test_log
            })

            if escalation_detected:
                correct_escalations += 1

        # Calculate escalation rate
        escalation_rate = correct_escalations / n_examples

        # Save detailed results if requested
        if save_results:
            output_path = os.path.join(
                self.output_dir,
                f"escalation_results_markers={context_markers}.json"
            )
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Saved detailed escalation results to {output_path}")

        return {
            "escalation_rate": escalation_rate,
            "n_examples": n_examples,
            "context_markers": context_markers
        }

    def evaluate_all(
        self,
        with_markers: bool = True,
        without_markers: bool = True,
        n_examples: int = 4,
        save_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Run all evaluations and compile results.

        Args:
            with_markers: Whether to test with context markers
            without_markers: Whether to test without context markers
            n_examples: Number of examples for each test
            save_summary: Whether to save the summary results

        Returns:
            Dictionary with all evaluation results
        """
        all_results = {}

        # Context retention evaluation
        if with_markers:
            all_results["context_retention_with_markers"] = self.evaluate_context_retention(
                n_examples=n_examples,
                context_markers=True
            )

        if without_markers:
            all_results["context_retention_without_markers"] = self.evaluate_context_retention(
                n_examples=n_examples,
                context_markers=False
            )

        # Consistency evaluation
        if with_markers:
            all_results["consistency_with_markers"] = self.evaluate_consistency(
                n_examples=n_examples,
                context_markers=True
            )

        if without_markers:
            all_results["consistency_without_markers"] = self.evaluate_consistency(
                n_examples=n_examples,
                context_markers=False
            )

        # Escalation evaluation
        if with_markers:
            all_results["escalation_with_markers"] = self.evaluate_escalation(
                n_examples=n_examples,
                context_markers=True
            )

        if without_markers:
            all_results["escalation_without_markers"] = self.evaluate_escalation(
                n_examples=n_examples,
                context_markers=False
            )

        # Calculate improvement from context markers
        if with_markers and without_markers:
            context_improvement = (
                all_results["context_retention_with_markers"]["context_retention_score"] -
                all_results["context_retention_without_markers"]["context_retention_score"]
            )

            consistency_improvement = (
                all_results["consistency_with_markers"]["consistency_score"] -
                all_results["consistency_without_markers"]["consistency_score"]
            )

            escalation_improvement = (
                all_results["escalation_with_markers"]["escalation_rate"] -
                all_results["escalation_without_markers"]["escalation_rate"]
            )

            all_results["improvement_summary"] = {
                "context_retention_improvement": context_improvement,
                "consistency_improvement": consistency_improvement,
                "escalation_improvement": escalation_improvement
            }

        # Save summary if requested
        if save_summary:
            output_path = os.path.join(self.output_dir, "evaluation_summary.json")
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)

            logger.info(f"Saved evaluation summary to {output_path}")

        return all_results

def compare_models(
    baseline_model: PreTrainedModel,
    baseline_tokenizer: AutoTokenizer,
    finetuned_model: PreTrainedModel,
    finetuned_tokenizer: AutoTokenizer,
    n_examples: int = 4,
    output_dir: str = "./model_comparison"
) -> Dict[str, Any]:
    """
    Compare baseline and fine-tuned models on medical conversation tasks.

    Args:
        baseline_model: Baseline model
        baseline_tokenizer: Baseline tokenizer
        finetuned_model: Fine-tuned model
        finetuned_tokenizer: Fine-tuned tokenizer
        n_examples: Number of examples for each test
        output_dir: Directory to save comparison results

    Returns:
        Dictionary with comparison results
    """
    # Create evaluators
    baseline_evaluator = MedicalConversationEvaluator(
        baseline_model,
        baseline_tokenizer,
        output_dir=os.path.join(output_dir, "baseline")
    )

    finetuned_evaluator = MedicalConversationEvaluator(
        finetuned_model,
        finetuned_tokenizer,
        output_dir=os.path.join(output_dir, "finetuned")
    )

    # Run evaluations
    logger.info("Evaluating baseline model...")
    baseline_results = baseline_evaluator.evaluate_all(
        n_examples=n_examples,
        with_markers=True,
        without_markers=False  # Only test with markers for efficiency
    )

    logger.info("Evaluating fine-tuned model...")
    finetuned_results = finetuned_evaluator.evaluate_all(
        n_examples=n_examples,
        with_markers=True,
        without_markers=False  # Only test with markers for efficiency
    )

    # Calculate improvements
    improvements = {
        "context_retention_improvement": (
            finetuned_results["context_retention_with_markers"]["context_retention_score"] -
            baseline_results["context_retention_with_markers"]["context_retention_score"]
        ),
        "consistency_improvement": (
            finetuned_results["consistency_with_markers"]["consistency_score"] -
            baseline_results["consistency_with_markers"]["consistency_score"]
        ),
        "escalation_improvement": (
            finetuned_results["escalation_with_markers"]["escalation_rate"] -
            baseline_results["escalation_with_markers"]["escalation_rate"]
        )
    }

    # Calculate overall improvement
    improvements["overall_improvement"] = (
        improvements["context_retention_improvement"] +
        improvements["consistency_improvement"] +
        improvements["escalation_improvement"]
    ) / 3.0

    # Compile comparison results
    comparison_results = {
        "baseline": baseline_results,
        "finetuned": finetuned_results,
        "improvements": improvements
    }

    # Save comparison results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2)

    logger.info(f"Saved model comparison results to {output_path}")

    return comparison_results
"""
Metrics for evaluating medical conversation quality.

This module provides specialized metrics for assessing context retention,
consistency, and escalation appropriateness in medical conversations.
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import torch
from datasets import Dataset
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Try to download NLTK resources if not already available
try:
    nltk.download('punkt', quiet=True)
except:
    logging.warning("Could not download NLTK resources. Sentence tokenization may be affected.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def extract_medical_entities(text: str, entity_list: List[str] = None) -> List[str]:
    """
    Extract medical entities from text using simple keyword matching.

    Args:
        text: Text to extract entities from
        entity_list: Optional list of medical entities to look for

    Returns:
        List of extracted entities
    """
    # Default list of common medical entities if none provided
    if entity_list is None:
        entity_list = [
            "pain", "ache", "fever", "cough", "headache", "nausea", "vomiting",
            "dizziness", "fatigue", "rash", "swelling", "inflammation", "bleeding",
            "infection", "shortness of breath", "weakness", "numbness", "diarrhea",
            "chest pain", "back pain", "sore throat", "runny nose", "congestion",
            "allergy", "medication", "antibiotic", "treatment", "doctor", "hospital"
        ]

    entities = []
    text_lower = text.lower()

    # Simple string matching approach
    for entity in entity_list:
        if entity.lower() in text_lower:
            entities.append(entity)

    return entities

def context_retention_score(
    reference_entities: List[str],
    response_text: str
) -> float:
    """
    Calculate context retention score by checking which reference entities
    are present in the response.

    Args:
        reference_entities: List of entities that should be remembered
        response_text: The response text to evaluate

    Returns:
        Score between 0.0 and 1.0
    """
    if not reference_entities:
        return 1.0  # No entities to retain

    found_entities = extract_medical_entities(response_text, reference_entities)

    # Calculate proportion of reference entities found in response
    score = len(found_entities) / len(reference_entities)

    return score

def response_consistency_score(
    responses: List[str],
    entity_extraction_fn: Callable = extract_medical_entities
) -> float:
    """
    Calculate consistency across multiple responses.

    Args:
        responses: List of response texts to compare
        entity_extraction_fn: Function to extract entities from responses

    Returns:
        Consistency score between 0.0 and 1.0
    """
    if len(responses) <= 1:
        return 1.0  # Single response is always consistent with itself

    # Extract entities from each response
    response_entities = [entity_extraction_fn(response) for response in responses]

    # Calculate Jaccard similarity between all pairs of entity sets
    scores = []
    for i in range(len(response_entities)):
        for j in range(i+1, len(response_entities)):
            set_i = set(response_entities[i])
            set_j = set(response_entities[j])

            if set_i or set_j:  # Avoid division by zero
                jaccard = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                scores.append(jaccard)

    # Average across all pairs
    return sum(scores) / len(scores) if scores else 0.0

def topic_shift_detection(
    turns: List[str],
    threshold: float = 0.3
) -> List[int]:
    """
    Detect shifts in conversation topic across turns.

    Args:
        turns: List of conversation turns
        threshold: Threshold for detecting a topic shift

    Returns:
        List of indices where topic shifts occur
    """
    # Implementation requires embedding model, using simpler approach for now
    if len(turns) <= 1:
        return []

    shift_indices = []
    prev_entities = extract_medical_entities(turns[0])

    for i in range(1, len(turns)):
        curr_entities = extract_medical_entities(turns[i])

        # Calculate Jaccard similarity between consecutive turns
        if prev_entities or curr_entities:
            overlap = len(set(prev_entities).intersection(set(curr_entities)))
            union = len(set(prev_entities).union(set(curr_entities)))
            similarity = overlap / union if union > 0 else 1.0

            # Detect shift if similarity is below threshold
            if similarity < threshold:
                shift_indices.append(i)

        prev_entities = curr_entities

    return shift_indices

def urgency_level_detection(text: str) -> str:
    """
    Detect the urgency level in a medical response.

    Args:
        text: Response text to analyze

    Returns:
        Urgency level: "low", "medium", "high", or "emergency"
    """
    text_lower = text.lower()

    # Emergency keywords
    emergency_terms = [
        "emergency", "call 911", "ambulance", "immediately", "right away",
        "urgent care", "emergency room", "er", "life-threatening"
    ]

    # High urgency keywords
    high_urgency_terms = [
        "see a doctor", "medical attention", "consult physician",
        "should not wait", "concerning", "worrisome", "as soon as possible"
    ]

    # Medium urgency keywords
    medium_urgency_terms = [
        "monitor", "watch for changes", "follow up", "if it persists",
        "within a few days", "contact your doctor if"
    ]

    # Count keyword matches for each level
    emergency_count = sum(term in text_lower for term in emergency_terms)
    high_count = sum(term in text_lower for term in high_urgency_terms)
    medium_count = sum(term in text_lower for term in medium_urgency_terms)

    # Determine urgency level based on keyword counts
    if emergency_count > 0:
        return "emergency"
    elif high_count > medium_count:
        return "high"
    elif medium_count > 0:
        return "medium"
    else:
        return "low"

def escalation_detection(
    initial_response: str,
    final_response: str
) -> Tuple[str, str, bool]:
    """
    Detect if there's an escalation in urgency between responses.

    Args:
        initial_response: Initial response to mild symptoms
        final_response: Final response after symptoms worsen

    Returns:
        Tuple of (initial_urgency, final_urgency, escalation_detected)
    """
    urgency_levels = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "emergency": 3
    }

    initial_urgency = urgency_level_detection(initial_response)
    final_urgency = urgency_level_detection(final_response)

    # Check if urgency increased
    escalation_detected = urgency_levels[final_urgency] > urgency_levels[initial_urgency]

    return initial_urgency, final_urgency, escalation_detected

def symptom_acknowledgment_rate(
    symptoms: List[str],
    response: str
) -> float:
    """
    Calculate the rate at which symptoms are acknowledged in the response.

    Args:
        symptoms: List of symptoms mentioned by the patient
        response: Doctor's response

    Returns:
        Acknowledgment rate between 0.0 and 1.0
    """
    if not symptoms:
        return 1.0  # No symptoms to acknowledge

    acknowledged = 0
    response_lower = response.lower()

    for symptom in symptoms:
        if symptom.lower() in response_lower:
            acknowledged += 1

    return acknowledged / len(symptoms)

def appropriate_action_rate(
    conditions: List[Dict[str, Any]],
    responses: List[str]
) -> float:
    """
    Calculate the rate at which responses recommend appropriate actions.

    Args:
        conditions: List of dictionaries with condition info including severity
        responses: Corresponding responses for each condition

    Returns:
        Appropriate action rate between 0.0 and 1.0
    """
    if not conditions or len(conditions) != len(responses):
        return 0.0

    appropriate_count = 0

    for i, condition in enumerate(conditions):
        severity = condition.get("severity", "medium")
        response = responses[i]
        urgency = urgency_level_detection(response)

        # Check if urgency matches severity
        if (severity == "mild" and urgency in ["low", "medium"]) or \
           (severity == "moderate" and urgency in ["medium", "high"]) or \
           (severity == "severe" and urgency in ["high", "emergency"]) or \
           (severity == "emergency" and urgency == "emergency"):
            appropriate_count += 1

    return appropriate_count / len(conditions)

def medical_advice_quality_score(response: str) -> Dict[str, float]:
    """
    Evaluate the quality of medical advice in a response.

    Args:
        response: Response to evaluate

    Returns:
        Dictionary with quality metrics
    """
    # Count of disclaimer/qualification statements
    qualification_terms = [
        "consult", "doctor", "physician", "healthcare provider", "medical professional",
        "not medical advice", "not a diagnosis", "cannot diagnose", "recommend seeing"
    ]
    qualification_score = min(1.0, sum(term in response.lower() for term in qualification_terms) / 3)

    # Check for evidence-based statements
    evidence_terms = [
        "studies show", "research indicates", "evidence suggests", "according to",
        "typically", "commonly", "generally", "most cases", "medical literature"
    ]
    evidence_score = min(1.0, sum(term in response.lower() for term in evidence_terms) / 2)

    # Check for comprehensive information
    sentences = sent_tokenize(response)
    comprehensiveness_score = min(1.0, len(sentences) / 5)

    # Check for actionable advice
    action_terms = [
        "try", "consider", "take", "apply", "use", "avoid", "limit", "increase", "decrease",
        "should", "monitor", "watch for", "if you experience", "call", "seek"
    ]
    actionable_score = min(1.0, sum(term in response.lower() for term in action_terms) / 3)

    # Combine scores
    quality_score = (qualification_score + evidence_score + comprehensiveness_score + actionable_score) / 4

    return {
        "qualification_score": qualification_score,
        "evidence_score": evidence_score,
        "comprehensiveness_score": comprehensiveness_score,
        "actionable_score": actionable_score,
        "overall_quality_score": quality_score
    }

def compute_metrics_for_dataset(
    dataset: Dataset,
    tokenizer,
    prompt_column: str = "prompt",
    response_column: str = "response",
    reference_column: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute metrics for a dataset of conversation examples.

    Args:
        dataset: Dataset with prompts and responses
        tokenizer: Tokenizer for text processing
        prompt_column: Column name for prompts
        response_column: Column name for responses
        reference_column: Optional column name for reference responses

    Returns:
        Dictionary of aggregate metrics
    """
    metrics = {
        "context_retention": [],
        "advice_quality": [],
        "symptom_acknowledgment": []
    }

    for example in dataset:
        prompt = example[prompt_column]
        response = example[response_column]

        # Extract patient symptoms from prompt
        symptoms = extract_medical_entities(prompt)

        # Calculate metrics
        retention = context_retention_score(symptoms, response)
        quality = medical_advice_quality_score(response)["overall_quality_score"]
        acknowledgment = symptom_acknowledgment_rate(symptoms, response)

        metrics["context_retention"].append(retention)
        metrics["advice_quality"].append(quality)
        metrics["symptom_acknowledgment"].append(acknowledgment)

    # Aggregate metrics
    aggregate_metrics = {
        "avg_context_retention": np.mean(metrics["context_retention"]),
        "avg_advice_quality": np.mean(metrics["advice_quality"]),
        "avg_symptom_acknowledgment": np.mean(metrics["symptom_acknowledgment"]),
        "examples_evaluated": len(dataset)
    }

    return aggregate_metrics

def evaluate_conversation_flow(
    conversation: List[Dict[str, str]],
    role_key: str = "role",
    content_key: str = "content"
) -> Dict[str, float]:
    """
    Evaluate the flow of a multi-turn conversation.

    Args:
        conversation: List of conversation turns with role and content
        role_key: Key for the role field in conversation turns
        content_key: Key for the content field in conversation turns

    Returns:
        Dictionary of flow metrics
    """
    if len(conversation) < 2:
        return {
            "turn_coherence": 1.0,
            "topic_maintenance": 1.0,
            "context_usage": 0.0
        }

    # Extract content for each turn
    turn_contents = [turn[content_key] for turn in conversation]

    # Detect topic shifts
    topic_shifts = topic_shift_detection(turn_contents)
    topic_maintenance = 1.0 - (len(topic_shifts) / (len(conversation) - 1)) if len(conversation) > 1 else 1.0

    # Calculate coherence between adjacent turns
    coherence_scores = []
    for i in range(1, len(conversation)):
        prev_entities = extract_medical_entities(turn_contents[i-1])
        curr_entities = extract_medical_entities(turn_contents[i])

        if prev_entities or curr_entities:
            set_prev = set(prev_entities)
            set_curr = set(curr_entities)
            jaccard = len(set_prev.intersection(set_curr)) / len(set_prev.union(set_curr)) if set_prev.union(set_curr) else 1.0
            coherence_scores.append(jaccard)

    # Calculate context usage across non-adjacent turns
    context_usage_scores = []
    all_patient_entities = set()

    for i, turn in enumerate(conversation):
        if turn[role_key].lower() == "patient":
            # Extract entities from patient turns
            entities = extract_medical_entities(turn[content_key])
            all_patient_entities.update(entities)
        elif i > 0 and turn[role_key].lower() == "doctor":
            # Check if doctor's turn uses context from any previous patient turn
            if all_patient_entities:
                found_entities = extract_medical_entities(turn[content_key], list(all_patient_entities))
                context_usage = len(found_entities) / len(all_patient_entities) if all_patient_entities else 0.0
                context_usage_scores.append(context_usage)

    return {
        "turn_coherence": np.mean(coherence_scores) if coherence_scores else 1.0,
        "topic_maintenance": topic_maintenance,
        "context_usage": np.mean(context_usage_scores) if context_usage_scores else 0.0
    }

def compute_all_metrics(
    conversation: List[Dict[str, str]],
    role_key: str = "role",
    content_key: str = "content"
) -> Dict[str, Any]:
    """
    Compute all available metrics for a medical conversation.

    Args:
        conversation: List of conversation turns with role and content
        role_key: Key for the role field in conversation turns
        content_key: Key for the content field in conversation turns

    Returns:
        Dictionary with all metrics
    """
    # Extract patient symptoms
    patient_turns = [turn for turn in conversation if turn[role_key].lower() == "patient"]
    doctor_turns = [turn for turn in conversation if turn[role_key].lower() == "doctor"]

    all_patient_symptoms = []
    for turn in patient_turns:
        symptoms = extract_medical_entities(turn[content_key])
        all_patient_symptoms.extend(symptoms)

    # Evaluate conversation flow
    flow_metrics = evaluate_conversation_flow(conversation, role_key, content_key)

    # Calculate context retention
    context_retention = []
    for turn in doctor_turns:
        score = context_retention_score(all_patient_symptoms, turn[content_key])
        context_retention.append(score)

    # Calculate advice quality
    quality_scores = []
    for turn in doctor_turns:
        quality = medical_advice_quality_score(turn[content_key])
        quality_scores.append(quality["overall_quality_score"])

    # Combine all metrics
    all_metrics = {
        "flow_metrics": flow_metrics,
        "avg_context_retention": np.mean(context_retention) if context_retention else 0.0,
        "avg_advice_quality": np.mean(quality_scores) if quality_scores else 0.0,
        "symptom_count": len(set(all_patient_symptoms)),
        "turn_count": len(conversation)
    }

    return all_metrics
