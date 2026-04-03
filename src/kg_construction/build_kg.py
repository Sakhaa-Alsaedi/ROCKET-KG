"""
KGBuilder — Multi-source Knowledge Graph construction pipeline.

Generates KG triples for every medical code using an LLM (GPT-4)
and writes tab-separated triple files:

    graphs/<type>/<ontology>/<code>.txt
    format per line:  head\trelation\ttail

Supports automatic resume: codes that already have files with ≥ min_lines
are skipped.

Usage::

    from src.kg_construction import KGBuilder

    builder = KGBuilder(
        openai_api_key="<ADD_YOUR_OPENAI_KEY_HERE>",  # ← Add your OpenAI API key here
        graphs_dir="/path/to/graphs",
        model="gpt-3.5-turbo",
    )
    builder.build_from_csv(
        csv_path="resources/CCSCM.csv",
        code_col="code",
        name_col="name",
        entity_type="condition",
        ontology="CCSCM",
    )
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_TRIPLE_PROMPT = """\
You are a biomedical knowledge graph expert.
Generate exactly {n_triples} factual knowledge graph triples for:

  Concept: {name} ({code})
  Type: {entity_type}

Rules:
1. Format: [ENTITY 1, RELATIONSHIP, ENTITY 2]
2. Use medical terminology; relationships should be clinically meaningful.
3. Include: causes, symptoms, treatments, comorbidities, risk factors, tests.
4. Each triple on its own line.

Examples:
[hypertension, is a risk factor for, stroke]
[metformin, treats, type 2 diabetes]
[troponin, is a biomarker for, myocardial infarction]

Generate {n_triples} triples now:
"""


def _parse_triples(text: str) -> List[Tuple[str, str, str]]:
    """Extract (head, relation, tail) tuples from LLM response."""
    triples = []
    pattern = re.compile(r"\[([^\[\]]+),\s*([^\[\]]+),\s*([^\[\]]+)\]")
    for match in pattern.finditer(text):
        h, r, t = (x.strip().lower() for x in match.groups())
        if h and r and t:
            triples.append((h, r, t))
    return triples


# ---------------------------------------------------------------------------
# KGBuilder class
# ---------------------------------------------------------------------------

class KGBuilder:
    """Build KG triple files from medical code dictionaries.

    Args:
        openai_api_key : OpenAI API key.  If None, reads ``OPENAI_API_KEY`` env.
        graphs_dir     : Root directory for triple output files.
        model          : OpenAI chat model to use.
        n_triples      : Target triples per concept.
        min_lines      : Skip files that already have ≥ min_lines triples.
        retry_delay    : Seconds to wait after rate-limit errors.
        max_retries    : Maximum API retries per concept.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        graphs_dir: str = "data/rocket_kg/graphs",
        model: str = "gpt-3.5-turbo",
        n_triples: int = 100,
        min_lines: int = 50,
        retry_delay: float = 5.0,
        max_retries: int = 3,
    ):
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.graphs_dir = Path(graphs_dir)
        self.model = model
        self.n_triples = n_triples
        self.min_lines = min_lines
        self.retry_delay = retry_delay
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_from_csv(
        self,
        csv_path: str,
        code_col: str,
        name_col: str,
        entity_type: str,
        ontology: str,
    ) -> int:
        """Generate KG triples for all codes in a CSV file.

        Args:
            csv_path   : Path to the code dictionary CSV.
            code_col   : Column name containing the code.
            name_col   : Column name containing the description.
            entity_type: "condition", "procedure", or "drug".
            ontology   : Ontology name (e.g. "CCSCM", "CCSPROC", "ATC3").

        Returns:
            Number of newly generated triple files.
        """
        out_dir = self.graphs_dir / entity_type / ontology
        out_dir.mkdir(parents=True, exist_ok=True)

        codes = self._load_csv(csv_path, code_col, name_col)
        logger.info(f"Loaded {len(codes)} codes from {csv_path}")

        generated = 0
        for code, name in codes:
            out_file = out_dir / f"{code}.txt"
            if self._should_skip(out_file):
                continue
            triples = self._generate_triples(code, name, entity_type)
            self._write_triples(out_file, triples)
            generated += 1
            logger.info(f"  [{generated}] {code}: {len(triples)} triples")

        return generated

    def build_from_dict(
        self,
        code_dict: dict,
        entity_type: str,
        ontology: str,
    ) -> int:
        """Generate KG triples from a {code: name} dictionary.

        Args:
            code_dict  : Mapping of code → name.
            entity_type: "condition", "procedure", or "drug".
            ontology   : Ontology name.

        Returns:
            Number of newly generated files.
        """
        out_dir = self.graphs_dir / entity_type / ontology
        out_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        for code, name in code_dict.items():
            out_file = out_dir / f"{code}.txt"
            if self._should_skip(out_file):
                continue
            triples = self._generate_triples(str(code), str(name), entity_type)
            self._write_triples(out_file, triples)
            generated += 1

        return generated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_csv(self, csv_path: str, code_col: str, name_col: str) -> List[Tuple[str, str]]:
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get(code_col, "").strip()
                name = row.get(name_col, "").strip()
                if code and name:
                    rows.append((code, name))
        return rows

    def _should_skip(self, path: Path) -> bool:
        if path.exists():
            with open(path) as f:
                n = sum(1 for line in f if line.strip())
            if n >= self.min_lines:
                return True
        return False

    def _generate_triples(
        self, code: str, name: str, entity_type: str
    ) -> List[Tuple[str, str, str]]:
        """Call OpenAI API and parse response into triples."""
        prompt = _TRIPLE_PROMPT.format(
            n_triples=self.n_triples, name=name, code=code, entity_type=entity_type
        )
        for attempt in range(self.max_retries):
            try:
                response = self._call_openai(prompt)
                triples = _parse_triples(response)
                if triples:
                    return triples
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {code}: {e}")
                time.sleep(self.retry_delay)
        return []

    def _call_openai(self, prompt: str) -> str:
        """Make an OpenAI chat completion request."""
        try:
            import openai
            openai.api_key = self.api_key
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            return resp["choices"][0]["message"]["content"]
        except ImportError:
            raise ImportError(
                "openai package required. pip install openai==0.27.4"
            )

    def _write_triples(self, path: Path, triples: List[Tuple[str, str, str]]):
        """Write triples to a tab-separated file."""
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")
