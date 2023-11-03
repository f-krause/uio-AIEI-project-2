from dataclasses import dataclass
import json


@dataclass
class Config:
    # Main mode
    mode: str = "prediction"  # prediction or classification

    # Data
    test_size: float = 0.2
    val_size: float = 0.1
    shuffle: bool = True
    seed: int = 42

    # Model config
    model: str = "LSTM"  # LSTM or RNN
    input_dim: int = 1
    hidden_dim: int = 32
    output_dim: int = 1
    num_layers: int = 1
    dropout: float = 0.0

    # Training config
    epochs: int = 1000
    lr: float = 0.01
    batch_size: int = 128
    eval_steps: int = 50

    def __post_init__(self):
        if self.mode.lower() == "classification":
            self.input_dim = 96
            self.hidden_dim = 128
            self.output_dim = 10

    def __str__(self):
        attributes = ["CONFIG:"] + [f"{name}: {value}" for name, value in self.__dict__.items()]
        return "\n  ".join(attributes)

    def save(self, save_name):
        path = f"config/config_{save_name}.json"
        with open(path, 'w') as json_file:
            json.dump(self.__dict__, json_file, indent=4)
        print(f"Config saved in: {path}")


def load_train_config(config_name) -> Config:
    path = f"config/{config_name}.json"
    with open(path, 'r') as json_file:
        json_data = json.load(json_file)
    config = Config(**json_data)
    print(f"Config loaded from: {path}")
    return config
