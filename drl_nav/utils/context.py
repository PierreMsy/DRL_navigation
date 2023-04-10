from dataclasses import dataclass

@dataclass
class Context:

    state_size: int
    action_size: int
    input_type: str