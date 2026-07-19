from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass


@dataclass
class TrainingPair:
    question: str
    code: str
    signal_type: str
    salestools_version: str
    dataset_seed: int
    paraphrase_id: int
    verified: bool = False
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_jsonl(cls, line: str) -> TrainingPair:
        data = json.loads(line.strip())
        return cls(**data)

    def to_chatml(self, system_prompt: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.code},
            ]
        }
