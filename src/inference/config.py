from dataclasses import dataclass
from pathlib import Path

@dataclass
class InferenceConfig:
    backend_directory = "../data/backend/"
    multipart_input_subdirectory = "multipart_input"
    multipart_output_subdirectory = "multipart_output"

    def get_multipart_input_dir(self) -> Path:
        return Path(self.backend_directory) / self.multipart_input_subdirectory

    def get_multipart_output_dir(self) -> Path:
        return Path(self.backend_directory) / self.multipart_output_subdirectory