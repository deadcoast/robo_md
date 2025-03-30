import numpy as np
from pathlib import Path
from typing import Dict, Any

from dataclasses import dataclass

from src.config.EngineConfig import SystemConfig
from src.core.NLPCore import NLPCore
from src.DataHandler import DataHandler
from src.ProcessingPool import ProcessingPool
from src.MarkdownProcessor import MarkdownProcessor

class EnhancedMarkdownProcessor(MarkdownProcessor):
    def __init__(self, config: SystemConfig):
        super().__init__(config)
        self.nlp_core = NLPCore()
        self.data_handler = DataHandler()

    async def process_vault(self, vault_path: Path) -> Dict[str, Any]:
        try:
            files = list(vault_path.rglob("*.md"))

            async with ProcessingPool() as pool:
                results = await pool.map(self._enhanced_file_processing, files)

            return self._aggregate_enhanced_results(results)

        except Exception as e:
            self.logger.error(f"Enhanced processing error: {str(e)}")
            raise EnhancedProcessingError(str(e)) from e

    async def _enhanced_file_processing(self, file_path: Path) -> Dict[str, Any]:
        content = await self._read_file(file_path)

        # Enhanced processing pipeline
        processed_content = await self.nlp_core.process_content(content)
        structured_data = await self.data_handler.process_data(processed_content)

        return {
            "path": str(file_path),
            "nlp_features": processed_content,
            "structured_data": structured_data,
            "metadata": self._extract_enhanced_metadata(content),
        }


@dataclass
class EnhancedFeatureSet:
    embeddings: np.ndarray
    topic_features: np.ndarray
    graph_features: np.ndarray
    metadata_features: np.ndarray
