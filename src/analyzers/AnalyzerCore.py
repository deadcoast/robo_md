"""Analyzer Core."""

from typing import Any

from src.AnalyzerCore import ResultAnalyzer


class AnalyzerCore:
    def __init__(self, analyzer: ResultAnalyzer):
        self.analyzer = analyzer

    async def analyze(self, data: Any) -> Any:
        return self.analyzer.analyze(data)

    async def validate(self, data: Any) -> Any:
        return self.analyzer.validate(data)

    async def compile(self, data: Any) -> Any:
        return self.analyzer.compile(data)

    async def execute(self, data: Any) -> Any:
        return self.analyzer.execute(data)

    async def finalize(self, data: Any) -> Any:
        return self.analyzer.finalize(data)
