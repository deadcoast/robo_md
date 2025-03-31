"""
Core natural language processing functionality.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import markdown
import spacy
import torch

from src.config.EngineConfig import SystemConfig


@dataclass
class NLPCore:
    config: SystemConfig

    def __init__(self, config: SystemConfig):
        self.config = config

    async def process_content(self, content: str) -> Dict[str, Any]:
        # Placeholder for actual NLP processing
        return {"processed_content": content}

    def _initialize(self):
        """
        Initializes the NLP core.

        This method is responsible for initializing the NLP core,
        setting up the necessary components and resources required for
        natural language processing.
        """
        self._load_nlp_model(torch)
        self._load_nlp_model(spacy)
        self._load_nlp_model(markdown)

    def _load_nlp_model(self, library_module):
        """
        Loads the NLP model.

        This method is responsible for loading the NLP model,
        setting up the necessary components and resources required for
        natural language processing.

        Args:
            library_module: The library module to use for loading the model.

        Returns:
            None
        """
        if library_module == spacy:
            # Load spaCy model based on configuration
            model_name = (
                self.config.nlp_model_name
                if hasattr(self.config, "nlp_model_name")
                else "en_core_web_sm"
            )
            try:
                self.spacy_nlp = spacy.load(model_name)
                print(f"Loaded spaCy model: {model_name}")
            except OSError:
                # Fallback to smaller model if the specified one isn't available
                print(
                    f"Model {model_name} not found. Downloading en_core_web_sm instead."
                )
                spacy.cli.download("en_core_web_sm")
                self.spacy_nlp = spacy.load("en_core_web_sm")

        elif library_module == torch:
            # Initialize PyTorch-based models if specified in config
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                and hasattr(self.config, "use_gpu")
                and self.config.use_gpu
                else "cpu"
            )
            print(f"PyTorch using device: {self.device}")

            # Initialize empty models dictionary to store any loaded models
            self.torch_models = {}

            # Load transformer models if specified in config
            if hasattr(self.config, "transformer_model_name"):
                from transformers import AutoModel, AutoTokenizer

                model_name = self.config.transformer_model_name
                try:
                    self.torch_models["tokenizer"] = AutoTokenizer.from_pretrained(
                        model_name
                    )
                    self.torch_models["model"] = AutoModel.from_pretrained(
                        model_name
                    ).to(self.device)
                    print(f"Loaded transformer model: {model_name}")
                except Exception as e:
                    print(f"Failed to load transformer model {model_name}: {str(e)}")

        elif library_module == markdown:
            # No actual model to load for markdown, just set it up
            self.markdown_extensions = (
                self.config.markdown_extensions
                if hasattr(self.config, "markdown_extensions")
                else ["extra", "smarty"]
            )

    def _read_file(self, file_path: Path) -> str:
        """
        Reads a file and returns its content.

        This method is responsible for reading a file and returning its content.

        Args:
            file_path: The path to the file to read.

        Returns:
            The content of the file as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read due to permission issues.
            UnicodeDecodeError: If the file cannot be decoded as UTF-8.
        """
        try:
            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Try to read the file with UTF-8 encoding
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Process markdown content if the file has a markdown extension
            if file_path.suffix.lower() in [".md", ".markdown"]:
                content = markdown.markdown(
                    content, extensions=getattr(self, "markdown_extensions", ["extra"])
                )

            return content
        except UnicodeDecodeError:
            # Fallback to read as binary if UTF-8 decoding fails
            try:
                with open(file_path, "rb") as file:
                    binary_content = file.read()

                # Try different encodings
                for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        return binary_content.decode(encoding)
                    except UnicodeDecodeError:
                        continue

                # Last resort: ignore errors
                return binary_content.decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                raise

    def _extract_enhanced_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced metadata from content.

        This method is responsible for extracting enhanced metadata from content.
        """
        # TODO: Implement the torch metadata extraction
        pass

    def _process_content(self, content: str) -> Dict[str, Any]:
        """
        Processes content using the NLP model.

        This method is responsible for processing content using the NLP model.
        """
        # TODO: Implement the torch NLP processing
        pass

    def _aggregate_enhanced_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregates enhanced results.

        This method is responsible for aggregating enhanced results.
        """
        # TODO: Implement the torch result aggregation
        pass

    def _extract_enhanced_features(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced features from content.

        This method is responsible for extracting enhanced features from content.
        """
        # TODO: Implement the torch feature extraction
        pass

    def _extract_enhanced_embeddings(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced embeddings from content.

        This method is responsible for extracting enhanced embeddings from content.
        """
        # TODO: Implement the torch embedding extraction
        pass

    def _extract_enhanced_topic_features(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced topic features from content.

        This method is responsible for extracting enhanced topic features from content.
        Uses NLP techniques to identify main topics and themes in the text.

        Args:
            content: The text content to extract topic features from.

        Returns:
            A dictionary containing topic features, including main topics,
            keywords, and topic distributions.
        """
        if not hasattr(self, "spacy_nlp"):
            # Initialize spaCy if not already done
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        keywords = [
            {"text": token.text, "pos": token.pos_, "count": 1}
            for token in doc
            if (
                token.pos_ in ["NOUN", "PROPN"]
                and not token.is_stop
                and len(token.text) > 1
            )
        ]
        # Aggregate and count keywords
        keyword_counts = {}
        for kw in keywords:
            text = kw["text"].lower()
            if text in keyword_counts:
                keyword_counts[text]["count"] += 1
            else:
                keyword_counts[text] = {"text": text, "pos": kw["pos"], "count": 1}

        # Sort keywords by frequency
        sorted_keywords = sorted(
            keyword_counts.values(), key=lambda x: x["count"], reverse=True
        )[
            :20
        ]  # Limit to top 20 keywords

        # Get named entities as potential topics
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # Extract noun phrases as potential topics
        noun_phrases = [
            chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1
        ]

        # Use TF-IDF like approach for topic scoring if we have transformer models loaded
        topic_scores = {}
        if hasattr(self, "torch_models") and "model" in self.torch_models:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Use paragraphs as documents for TF-IDF
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if len(paragraphs) > 1:
                try:
                    self._extracted_from__extract_enhanced_topic_features_(
                        TfidfVectorizer, paragraphs, topic_scores
                    )
                except Exception as e:
                    print(f"TF-IDF calculation failed: {str(e)}")

        return {
            "keywords": sorted_keywords,
            "entities": entities,
            "noun_phrases": noun_phrases[:20],  # Limit to top 20 phrases
            "topic_scores": topic_scores,
            "main_topics": sorted_keywords[:5],  # Top 5 keywords as main topics
        }

    # TODO Rename this here and in `_extract_enhanced_topic_features`
    def _extracted_from__extract_enhanced_topic_features_(
        self, TfidfVectorizer, paragraphs, topic_scores
    ):
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        feature_names = vectorizer.get_feature_names_out()

        # Compute average TF-IDF scores across documents
        import numpy as np

        tfidf_means = np.mean(tfidf_matrix.toarray(), axis=0)

        # Create topic scores
        for i, feature in enumerate(feature_names):
            topic_scores[feature] = float(tfidf_means[i])

    def _extract_enhanced_graph_features(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced graph features from content.

        This method is responsible for extracting enhanced graph features from content.
        Creates a graph representation of the text based on semantic relationships.

        Args:
            content: The text content to extract graph features from.

        Returns:
            A dictionary containing graph features, including nodes, edges,
            centrality measures, and graph statistics.
        """
        if not hasattr(self, "spacy_nlp"):
            # Initialize spaCy if not already done
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        # Create a graph representation using networkx
        import networkx as nx

        G = nx.Graph()

        # Add entities as nodes
        for ent in doc.ents:
            if not G.has_node(ent.text):
                G.add_node(ent.text, type="entity", label=ent.label_)

        # Add important tokens as nodes (nouns, verbs, adjectives)
        for token in doc:
            if (
                token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]
                and not token.is_stop
                and not G.has_node(token.text)
            ):
                G.add_node(token.text, type="token", pos=token.pos_)

        # Add edges based on sentence co-occurrence
        for sent in doc.sents:
            sent_entities = list(sent.ents)
            sent_tokens = [
                token
                for token in sent
                if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop
            ]

            # Connect entities that appear in the same sentence
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i + 1 :]:
                    if G.has_node(ent1.text) and G.has_node(ent2.text):
                        if G.has_edge(ent1.text, ent2.text):
                            G[ent1.text][ent2.text]["weight"] += 1
                        else:
                            G.add_edge(
                                ent1.text, ent2.text, weight=1, type="entity_entity"
                            )

            # Connect tokens that appear in the same sentence
            for i, token1 in enumerate(sent_tokens):
                for token2 in sent_tokens[i + 1 :]:
                    if G.has_node(token1.text) and G.has_node(token2.text):
                        if G.has_edge(token1.text, token2.text):
                            G[token1.text][token2.text]["weight"] += 1
                        else:
                            G.add_edge(
                                token1.text, token2.text, weight=1, type="token_token"
                            )

        # Compute centrality measures
        centrality = {}
        if len(G.nodes) > 0:
            try:
                centrality["degree"] = nx.degree_centrality(G)
                if len(G.nodes) > 1:  # These require connected graphs
                    try:
                        centrality["betweenness"] = nx.betweenness_centrality(G)
                        centrality["closeness"] = nx.closeness_centrality(G)
                    except Exception as e:
                        print(f"Some centrality measures failed: {str(e)}")
            except Exception as e:
                print(f"Centrality calculation failed: {str(e)}")

        # Prepare output
        nodes = [
            {
                "id": node,
                "type": data.get("type", "unknown"),
                "metadata": {k: v for k, v in data.items() if k != "type"},
            }
            for node, data in G.nodes(data=True)
        ]

        edges = [
            {
                "source": u,
                "target": v,
                "weight": data.get("weight", 1),
                "type": data.get("type", "unknown"),
            }
            for u, v, data in G.edges(data=True)
        ]

        # Graph statistics
        stats = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G) if G.number_of_nodes() > 1 else 0,
            "components": (
                nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
            ),
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "centrality": centrality,
            "stats": stats,
            "graph_object": G,  # Return the graph object for further processing
        }

    def _extract_enhanced_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced metadata from content.

        This method is responsible for extracting enhanced metadata from content.
        Analyzes the text to extract information about its structure, language,
        readability, and other metadata.

        Args:
            content: The text content to extract metadata from.

        Returns:
            A dictionary containing metadata about the content, including length,
            language statistics, readability metrics, and content structure.
        """
        if not hasattr(self, "spacy_nlp"):
            # Initialize spaCy if not already done
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        # Basic text statistics
        char_count = len(content)
        word_count = len(
            [token for token in doc if not token.is_punct and not token.is_space]
        )
        sentence_count = len(list(doc.sents))
        paragraph_count = len([p for p in content.split("\n\n") if p.strip()])

        # Average word length
        avg_word_length = sum(
            len(token.text)
            for token in doc
            if not token.is_punct and not token.is_space
        ) / max(1, word_count)

        # Average sentence length
        avg_sentence_length = word_count / max(1, sentence_count)

        # Language detection
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # For consistent results
        try:
            language = detect(content)
        except Exception:  # Ignoring specific error details
            language = "unknown"

        # Readability scores
        import textstat

        readability = {
            "flesch_reading_ease": textstat.flesch_reading_ease(content),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(content),
            "gunning_fog": textstat.gunning_fog(content),
            "smog_index": textstat.smog_index(content),
            "coleman_liau_index": textstat.coleman_liau_index(content),
            "automated_readability_index": textstat.automated_readability_index(
                content
            ),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(
                content
            ),
        }

        # Part-of-speech distribution
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        # Named entity counts
        entity_counts = {}
        for ent in doc.ents:
            entity_type = ent.label_
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        # Content structure analysis
        structure = {
            "paragraphs": paragraph_count,
            "sentences": sentence_count,
            "words": word_count,
            "characters": char_count,
            "avg_paragraph_length": word_count / max(1, paragraph_count),
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
        }

        # Sentiment analysis
        from textblob import TextBlob

        blob = TextBlob(content)
        sentiment = {
            "polarity": blob.sentiment.polarity,  # -1.0 to 1.0 (negative to positive)
            "subjectivity": blob.sentiment.subjectivity,  # 0.0 to 1.0 (objective to subjective)
        }

        return {
            "structure": structure,
            "language": language,
            "readability": readability,
            "pos_distribution": pos_counts,
            "entity_counts": entity_counts,
            "sentiment": sentiment,
            "creation_time": (
                torch.cuda.current_stream()
                .record_event()
                .elapsed_time(torch.cuda.Event(enable_timing=True))
                if torch.cuda.is_available()
                else 0
            ),
        }

    def _extract_enhanced_stats(self, content: str) -> Dict[str, Any]:
        """
        Extracts enhanced statistics from content.

        This method is responsible for extracting enhanced statistics from content.
        Performs statistical analysis on the text to identify patterns and insights.

        Args:
            content: The text content to extract statistics from.

        Returns:
            A dictionary containing statistical analysis of the content, including
            word frequency distributions, n-gram analysis, and other statistical measures.
        """
        if not hasattr(self, "spacy_nlp"):
            # Initialize spaCy if not already done
            self._load_nlp_model(spacy)

        # Process the content with spaCy
        doc = self.spacy_nlp(content)

        # Extract words (excluding punctuation, spaces, and stop words)
        words = [
            token.text.lower()
            for token in doc
            if not token.is_punct and not token.is_space and not token.is_stop
        ]

        # Word frequency distribution
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_word_freq[:50]  # Top 50 words

        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        trigrams = [
            (words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)
        ]
        # Calculate bigram frequencies
        bigram_freq = {}
        for bg in bigrams:
            bigram_text = f"{bg[0]} {bg[1]}"
            bigram_freq[bigram_text] = bigram_freq.get(bigram_text, 0) + 1

        # Calculate trigram frequencies
        trigram_freq = {}
        for tg in trigrams:
            trigram_text = f"{tg[0]} {tg[1]} {tg[2]}"
            trigram_freq[trigram_text] = trigram_freq.get(trigram_text, 0) + 1

        # Sort n-grams by frequency
        sorted_bigram_freq = sorted(
            bigram_freq.items(), key=lambda x: x[1], reverse=True
        )
        sorted_trigram_freq = sorted(
            trigram_freq.items(), key=lambda x: x[1], reverse=True
        )

        top_bigrams = sorted_bigram_freq[:30]  # Top 30 bigrams
        top_trigrams = sorted_trigram_freq[:20]  # Top 20 trigrams

        # Calculate lexical diversity (unique words / total words)
        lexical_diversity = len(word_freq) / max(1, len(words))

        # Basic statistical measures
        word_lengths = [len(word) for word in words]
        from statistics import mean, median, stdev

        word_length_stats = {
            "mean": mean(word_lengths) if word_lengths else 0,
            "median": median(word_lengths) if word_lengths else 0,
            "std_dev": stdev(word_lengths) if len(word_lengths) > 1 else 0,
            "min": min(word_lengths, default=0),
            "max": max(word_lengths, default=0),
        }

        # Sentence length statistics
        sentence_lengths = [
            len([token for token in sent if not token.is_punct and not token.is_space])
            for sent in doc.sents
        ]
        sentence_length_stats = {
            "mean": mean(sentence_lengths) if sentence_lengths else 0,
            "median": median(sentence_lengths) if sentence_lengths else 0,
            "std_dev": stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0,
            "min": min(sentence_lengths, default=0),
            "max": max(sentence_lengths, default=0),
        }

        # Calculate TF-IDF if we're using PyTorch
        tfidf_analysis = {}
        if hasattr(self, "torch_models"):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                # Split content into paragraphs
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

                if len(paragraphs) > 1:
                    self._extracted_from__extract_enhanced_stats_110(
                        TfidfVectorizer, paragraphs, tfidf_analysis
                    )
            except Exception as e:
                print(f"TF-IDF analysis failed: {str(e)}")

        return {
            "word_frequency": dict(top_words),
            "bigram_frequency": dict(top_bigrams),
            "trigram_frequency": dict(top_trigrams),
            "lexical_diversity": lexical_diversity,
            "word_length_stats": word_length_stats,
            "sentence_length_stats": sentence_length_stats,
            "total_unique_words": len(word_freq),
            "total_words": len(words),
            "tfidf_analysis": tfidf_analysis,
        }

    # TODO Rename this here and in `_extract_enhanced_stats`
    def _extracted_from__extract_enhanced_stats_110(self, TfidfVectorizer, paragraphs, tfidf_analysis):
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        feature_names = vectorizer.get_feature_names_out()
        import numpy as np

        tfidf_means = np.mean(tfidf_matrix.toarray(), axis=0)

        # Get most significant terms by TF-IDF
        significant_terms = [
            (feature_names[i], float(tfidf_means[i]))
            for i in np.argsort(tfidf_means)[::-1][:50]
        ]
        tfidf_analysis["significant_terms"] = significant_terms
