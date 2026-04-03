"""
ROCKET Agent — ReAct loop for clinical knowledge retrieval.

Architecture (ReAct: Reason + Act):
  1. The agent receives a clinical question.
  2. It reasons about which tool to invoke.
  3. It acts by calling one of its registered tools.
  4. It observes the result and reasons again.
  5. Steps 2–4 repeat until a final answer is produced.

Available tools:
  - search_kg(query)        : Retrieve nearest-neighbour entities from the KG.
  - get_patient_history(pid): Return a patient's EHR summary.
  - get_drug_interactions(drugs): Return known DDI information.
  - compute_similarity(a, b): Cosine similarity between two medical terms.

The agent is LLM-agnostic.  Pass any callable that takes a prompt string
and returns a string as ``llm_fn``.

Usage::

    from src.agent import RocketAgent

    agent = RocketAgent(
        kg_entities=["sepsis", "acute kidney injury", ...],
        kg_embeddings=entity_emb_matrix,
        llm_fn=my_llm,           # callable: str → str
        max_steps=5,
        verbose=True,
    )
    answer = agent.run("What conditions are associated with high mortality in ICU patients?")
    print(answer)
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class KGSearchTool:
    """Nearest-neighbour search in the knowledge graph embedding space.

    Args:
        entities   : List of entity name strings.
        embeddings : [N, D] embedding matrix (L2-normalised recommended).
    """

    def __init__(self, entities: List[str], embeddings: np.ndarray):
        self.entities = entities
        emb = embeddings.astype(np.float64)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
        self.embeddings = emb / norms
        self._name_to_idx = {e: i for i, e in enumerate(entities)}

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k entities by cosine similarity.

        Args:
            query_emb: [D,] query embedding (will be normalised internally).
            top_k    : Number of results to return.

        Returns:
            List of (entity_name, similarity_score) tuples.
        """
        q = query_emb.astype(np.float64)
        q = q / (np.linalg.norm(q) + 1e-10)
        sims = self.embeddings @ q
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.entities[i], float(sims[i])) for i in top_idx]

    def get_embedding(self, entity: str) -> Optional[np.ndarray]:
        """Return embedding for a named entity."""
        idx = self._name_to_idx.get(entity)
        return self.embeddings[idx] if idx is not None else None


class PatientHistoryTool:
    """Returns a formatted EHR summary for a patient.

    Args:
        patient_records: Dict mapping patient_id → dict with keys
                         "conditions", "procedures", "drugs", "visits".
    """

    def __init__(self, patient_records: Optional[Dict] = None):
        self.records = patient_records or {}

    def get(self, patient_id: str) -> str:
        """Return a text summary of the patient's EHR."""
        rec = self.records.get(patient_id)
        if rec is None:
            return f"No record found for patient {patient_id}."
        lines = [f"Patient: {patient_id}"]
        if "conditions" in rec:
            lines.append(f"  Conditions : {', '.join(rec['conditions'][-5:])}")
        if "procedures" in rec:
            lines.append(f"  Procedures : {', '.join(rec['procedures'][-5:])}")
        if "drugs" in rec:
            lines.append(f"  Medications: {', '.join(rec['drugs'][-5:])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a clinical knowledge assistant with access to a medical knowledge graph.
Answer the clinical question step by step using ReAct format:

Thought: <your reasoning>
Action: <tool_name>(<arguments>)
Observation: <tool result — filled in by the system>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: <answer>

Available tools:
  - search_kg(query: str, top_k: int=5) — search knowledge graph for related entities
  - get_patient_history(patient_id: str) — retrieve patient EHR summary
  - compute_similarity(term_a: str, term_b: str) — compute semantic similarity

Always end with "Final Answer:".
"""


class RocketAgent:
    """ReAct agent for clinical question answering.

    Args:
        entities    : List of KG entity names.
        kg_embeddings: [N, D] KG entity embeddings.
        llm_fn      : Callable str→str.  Receives a prompt, returns LLM output.
        patient_records: Optional patient EHR dict.
        embed_fn    : Callable str→np.ndarray.  Embeds a query string.
                      If None, falls back to random vectors (testing only).
        max_steps   : Maximum ReAct iterations.
        verbose     : Print intermediate steps.
    """

    def __init__(
        self,
        entities: List[str],
        kg_embeddings: np.ndarray,
        llm_fn: Callable[[str], str],
        patient_records: Optional[Dict] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        max_steps: int = 6,
        verbose: bool = False,
    ):
        self.kg_search = KGSearchTool(entities, kg_embeddings)
        self.patient_history = PatientHistoryTool(patient_records)
        self.llm_fn = llm_fn
        self.embed_fn = embed_fn
        self.max_steps = max_steps
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, action_str: str) -> str:
        """Parse and execute an action string like tool_name(args)."""
        action_str = action_str.strip()
        match = re.match(r"(\w+)\((.*)\)$", action_str, re.DOTALL)
        if not match:
            return f"[Error] Could not parse action: {action_str!r}"

        tool_name, raw_args = match.group(1).strip(), match.group(2).strip()

        try:
            if tool_name == "search_kg":
                parts = [a.strip().strip("'\"") for a in raw_args.split(",")]
                query = parts[0]
                top_k = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                if self.embed_fn is not None:
                    q_emb = self.embed_fn(query)
                else:
                    # Testing fallback
                    q_emb = np.random.randn(self.kg_search.embeddings.shape[1])
                results = self.kg_search.search(q_emb, top_k=top_k)
                return "\n".join(f"  {ent}: {sim:.3f}" for ent, sim in results)

            elif tool_name == "get_patient_history":
                parts = [a.strip().strip("'\"") for a in raw_args.split(",")]
                pid = parts[0]
                return self.patient_history.get(pid)

            elif tool_name == "compute_similarity":
                parts = [a.strip().strip("'\"") for a in raw_args.split(",")]
                a, b = parts[0], parts[1]
                if self.embed_fn is not None:
                    emb_a = self.embed_fn(a)
                    emb_b = self.embed_fn(b)
                    emb_a = emb_a / (np.linalg.norm(emb_a) + 1e-10)
                    emb_b = emb_b / (np.linalg.norm(emb_b) + 1e-10)
                    sim = float(emb_a @ emb_b)
                else:
                    emb_a = self.kg_search.get_embedding(a)
                    emb_b = self.kg_search.get_embedding(b)
                    if emb_a is None or emb_b is None:
                        return f"One or both terms not found: {a!r}, {b!r}"
                    sim = float(emb_a @ emb_b)
                return f"Similarity({a!r}, {b!r}) = {sim:.4f}"

            else:
                return f"[Error] Unknown tool: {tool_name!r}"

        except Exception as e:
            return f"[Error] Tool {tool_name!r} raised: {e}"

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def run(self, question: str) -> str:
        """Run the ReAct loop to answer a clinical question.

        Args:
            question: Natural language clinical question.

        Returns:
            Final answer string.
        """
        prompt = _SYSTEM_PROMPT + f"\n\nQuestion: {question}\n"
        history = ""

        for step in range(self.max_steps):
            full_prompt = prompt + history
            response = self.llm_fn(full_prompt)

            if self.verbose:
                logger.info(f"[Step {step + 1}] Response:\n{response}")

            history += response + "\n"

            # Check for Final Answer
            if "Final Answer:" in response:
                match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)
                if match:
                    return match.group(1).strip()
                return response.split("Final Answer:")[-1].strip()

            # Extract Action and execute tool
            action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response)
            if action_match:
                action_str = action_match.group(1).strip()
                observation = self._execute_tool(action_str)
                if self.verbose:
                    logger.info(f"[Observation] {observation}")
                history += f"Observation: {observation}\n"
            else:
                # No action found — ask for clarification
                history += "Observation: No action found. Please specify an Action or provide a Final Answer.\n"

        return f"[Max steps reached] Last response: {response}"
