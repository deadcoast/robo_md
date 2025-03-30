"""Analyzer Core."""

from typing import Any

from src.AnalyzerCore import ResultAnalyzer


class AnalyzerCore:
    """
    Represents an analyzer core that handles the analysis of data.
    """

    def __init__(self, analyzer: ResultAnalyzer):
        """
        Initializes the analyzer core with a specific analyzer.

        :param analyzer: The analyzer to be used for analysis.
        :type analyzer: ResultAnalyzer
        """
        self.analyzer = analyzer

    async def analyze(self, data: Any) -> Any:
        """
        Analyzes the provided data.

        :param data: The data to be analyzed.
        :type data: Any
        :return: The result of the analysis.
        :rtype: Any
        """
        return self.analyzer.analyze(data)

    async def validate(self, data: Any) -> Any:
        """
        Validates the provided data.

        :param data: The data to be validated.
        :type data: Any
        :return: The result of the validation.
        :rtype: Any
        """
        return self.analyzer.validate(data)

    async def compile(self, data: Any) -> Any:
        """
        Compiles the provided data.

        :param data: The data to be compiled.
        :type data: Any
        :return: The result of the compilation.
        :rtype: Any
        """
        return self.analyzer.compile(data)

    async def execute(self, data: Any) -> Any:
        """
        Executes the provided data.

        :param data: The data to be executed.
        :type data: Any
        :return: The result of the execution.
        :rtype: Any
        """
        return self.analyzer.execute(data)

    async def finalize(self, data: Any) -> Any:
        """
        Finalizes the provided data.

        :param data: The data to be finalized.
        :type data: Any
        :return: The result of the finalization.
        :rtype: Any
        """
        return self.analyzer.finalize(data)
