"""MongoDB-backed store with vector search capabilities.

Extends LangGraph's BaseStore to provide persistent storage using MongoDB,
with optional semantic search through embeddings.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from python.env_utils import doublecheck_pkgs, doublecheck_env

# Load environment variables from .env
load_dotenv()

# Check and print results
doublecheck_env(".env")  # check environmental variables
doublecheck_pkgs(pyproject_path="pyproject.toml", verbose=True)

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

logger = logging.getLogger(__name__)


class MongoDBVectorStore(BaseStore):
    """MongoDB-backed store with optional vector search.

    Provides persistent storage using MongoDB with support for hierarchical
    namespaces, key-value operations, and semantic search via embeddings.

    Args:
        connection_string: MongoDB connection URI (e.g., "mongodb://localhost:27017")
        database: Name of the MongoDB database to use
        collection: Name of the collection to store documents in
        index: Optional IndexConfig for vector search configuration

    Example:
        Basic usage:
        ```python
        store = MongoDBVectorStore(
            connection_string="mongodb://localhost:27017",
            database="mydb",
            collection="store"
        )
        store.put(("users", "123"), "prefs", {"theme": "dark"})
        item = store.get(("users", "123"), "prefs")
        ```

        With vector search:
        ```python
        from langgraph.store.base import IndexConfig

        store = MongoDBVectorStore(
            connection_string="mongodb://localhost:27017",
            database="mydb",
            collection="store",
            index=IndexConfig(dims=1536, embed_model="text-embedding-3-small")
        )
        store.put(("docs",), "doc1", {"text": "Python tutorial"})
        results = store.search(("docs",), query="python programming")
        ```
    """

    __slots__ = (
        "_client",
        "_db",
        "_collection",
        "index_config",
        "embeddings",
    )

    def __init__(
        self,
        connection_string: str,
        database: str,
        collection: str,
        *,
        index: IndexConfig | None = None,
    ) -> None:
        self._client: MongoClient = MongoClient(connection_string)
        self._db: Database = self._client[database]
        self._collection: Collection = self._db[collection]

        # Create indexes for efficient querying
        self._collection.create_index([("namespace", ASCENDING)])
        self._collection.create_index([("_id", ASCENDING)])

        self.index_config = index
        if self.index_config:
            self.index_config = self.index_config.copy()
            self.embeddings: Embeddings | None = ensure_embeddings(
                self.index_config.get("embed"),
            )
            self.index_config["__tokenized_fields"] = [
                (p, tokenize_path(p)) if p != "$" else (p, p)
                for p in (self.index_config.get("fields") or ["$"])
            ]
        else:
            self.index_config = None
            self.embeddings = None

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations."""
        results, put_ops, search_ops = self._prepare_ops(ops)

        # Handle search operations with embeddings
        if search_ops:
            query_embeddings = self._embed_search_queries(search_ops)
            self._batch_search(search_ops, query_embeddings, results)

        # Handle put operations with embeddings
        to_embed = self._extract_texts(put_ops)
        if to_embed and self.index_config and self.embeddings:
            embeddings = self.embeddings.embed_documents(list(to_embed))
            self._apply_put_ops_with_embeddings(put_ops, to_embed, embeddings)
        else:
            self._apply_put_ops(put_ops)

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously."""
        results, put_ops, search_ops = self._prepare_ops(ops)

        # Handle search operations with embeddings
        if search_ops:
            query_embeddings = await self._aembed_search_queries(search_ops)
            self._batch_search(search_ops, query_embeddings, results)

        # Handle put operations with embeddings
        to_embed = self._extract_texts(put_ops)
        if to_embed and self.index_config and self.embeddings:
            embeddings = await self.embeddings.aembed_documents(list(to_embed))
            self._apply_put_ops_with_embeddings(put_ops, to_embed, embeddings)
        else:
            self._apply_put_ops(put_ops)

        return results

    def _serialize_namespace(self, namespace: tuple[str, ...]) -> str:
        """Convert namespace tuple to string for MongoDB _id."""
        return ":".join(namespace) if namespace else ""

    def _deserialize_namespace(self, namespace_str: str) -> tuple[str, ...]:
        """Convert namespace string back to tuple."""
        return tuple(namespace_str.split(":")) if namespace_str else ()

    def _make_doc_id(self, namespace: tuple[str, ...], key: str) -> str:
        """Create MongoDB document _id from namespace and key."""
        ns_str = self._serialize_namespace(namespace)
        return f"{ns_str}:{key}" if ns_str else key

    def _prepare_ops(
        self, ops: Iterable[Op]
    ) -> tuple[
        list[Result],
        dict[tuple[tuple[str, ...], str], PutOp],
        dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
    ]:
        """Prepare operations for batch execution."""
        results: list[Result] = []
        put_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        search_ops: dict[
            int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]
        ] = {}

        for i, op in enumerate(ops):
            if isinstance(op, GetOp):
                item = self._handle_get(op)
                results.append(item)
            elif isinstance(op, SearchOp):
                search_ops[i] = (op, self._filter_items(op))
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            elif isinstance(op, PutOp):
                put_ops[(op.namespace, op.key)] = op
                results.append(None)
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

        return results, put_ops, search_ops

    def _handle_get(self, op: GetOp) -> Item | None:
        """Retrieve a single item from MongoDB."""
        doc_id = self._make_doc_id(op.namespace, op.key)
        doc = self._collection.find_one({"_id": doc_id})

        if doc:
            return Item(
                value=doc["value"],
                key=doc["key"],
                namespace=tuple(doc["namespace"]),
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
            )
        return None

    def _filter_items(self, op: SearchOp) -> list[tuple[Item, list[list[float]]]]:
        """Filter items by namespace prefix and filter conditions."""
        namespace_prefix = op.namespace_prefix

        # Build MongoDB query
        query: dict[str, Any] = {}

        # Filter by namespace prefix
        if namespace_prefix:
            prefix_str = self._serialize_namespace(namespace_prefix)
            query["namespace"] = {
                "$elemMatch": {"$eq": prefix_str.split(":")} if prefix_str else {}
            }
            # Better approach: match namespace array prefix
            ns_array = list(namespace_prefix)
            if ns_array:
                query["namespace"] = {
                    "$gte": ns_array,
                    "$lt": ns_array[:-1] + [ns_array[-1] + "\uffff"]
                }

        # Apply field filters
        if op.filter:
            for key, filter_value in op.filter.items():
                query[f"value.{key}"] = self._build_filter_condition(filter_value)

        # Query MongoDB
        docs = list(self._collection.find(query))

        # Convert to items with embeddings
        filtered = []
        for doc in docs:
            item = Item(
                value=doc["value"],
                key=doc["key"],
                namespace=tuple(doc["namespace"]),
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
            )

            # Extract embeddings if present
            embeddings = []
            if op.query and "embedding" in doc and doc["embedding"]:
                if isinstance(doc["embedding"], dict):
                    embeddings = list(doc["embedding"].values())
                elif isinstance(doc["embedding"], list):
                    embeddings = [doc["embedding"]]

            filtered.append((item, embeddings))

        return filtered

    def _build_filter_condition(self, filter_value: Any) -> Any:
        """Build MongoDB filter condition from filter value."""
        if isinstance(filter_value, dict):
            # Handle operators like $eq, $ne, $gt, etc.
            return filter_value
        else:
            # Simple equality
            return {"$eq": filter_value}

    def _embed_search_queries(
        self,
        search_ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
    ) -> dict[str, list[float]]:
        """Generate embeddings for search queries."""
        query_embeddings = {}
        if self.index_config and self.embeddings and search_ops:
            queries = {op.query for (op, _) in search_ops.values() if op.query}

            if queries:
                for query in queries:
                    query_embeddings[query] = self.embeddings.embed_query(query)

        return query_embeddings

    async def _aembed_search_queries(
        self,
        search_ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
    ) -> dict[str, list[float]]:
        """Generate embeddings for search queries asynchronously."""
        query_embeddings = {}
        if self.index_config and self.embeddings and search_ops:
            queries = {op.query for (op, _) in search_ops.values() if op.query}

            if queries:
                coros = [self.embeddings.aembed_query(q) for q in list(queries)]
                results = await asyncio.gather(*coros)
                query_embeddings = dict(zip(queries, results, strict=False))

        return query_embeddings

    def _batch_search(
        self,
        ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
        query_embeddings: dict[str, list[float]],
        results: list[Result],
    ) -> None:
        """Perform batch similarity search for multiple queries."""
        for i, (op, candidates) in ops.items():
            if not candidates:
                results[i] = []
                continue

            if op.query and query_embeddings:
                query_embedding = query_embeddings[op.query]
                flat_items, flat_vectors = [], []
                scoreless = []

                for item, vectors in candidates:
                    for vector in vectors:
                        flat_items.append(item)
                        flat_vectors.append(vector)
                    if not vectors:
                        scoreless.append(item)

                # Compute cosine similarity scores
                scores = self._cosine_similarity(query_embedding, flat_vectors)
                sorted_results = sorted(
                    zip(scores, flat_items, strict=False),
                    key=lambda x: x[0],
                    reverse=True,
                )

                # Max pooling to deduplicate
                seen: set[tuple[tuple[str, ...], str]] = set()
                kept: list[tuple[float | None, Item]] = []

                for score, item in sorted_results:
                    key = (item.namespace, item.key)
                    if key in seen:
                        continue
                    ix = len(seen)
                    seen.add(key)
                    if ix >= op.offset + op.limit:
                        break
                    if ix < op.offset:
                        continue

                    kept.append((score, item))

                # Add scoreless items if needed
                if scoreless and len(kept) < op.limit:
                    kept.extend(
                        (None, item) for item in scoreless[: op.limit - len(kept)]
                    )

                results[i] = [
                    SearchItem(
                        namespace=item.namespace,
                        key=item.key,
                        value=item.value,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                        score=float(score) if score is not None else None,
                    )
                    for score, item in kept
                ]
            else:
                # No query, just return filtered items with pagination
                results[i] = [
                    SearchItem(
                        namespace=item.namespace,
                        key=item.key,
                        value=item.value,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                    )
                    for (item, _) in candidates[op.offset : op.offset + op.limit]
                ]

    def _cosine_similarity(self, X: list[float], Y: list[list[float]]) -> list[float]:
        """Compute cosine similarity between a vector X and matrix Y."""
        if not Y:
            return []

        X_arr = np.array(X)
        Y_arr = np.array(Y)

        # Normalize vectors
        X_norm = X_arr / np.linalg.norm(X_arr)
        Y_norm = Y_arr / np.linalg.norm(Y_arr, axis=1, keepdims=True)

        # Compute dot product (cosine similarity for normalized vectors)
        similarities = np.dot(Y_norm, X_norm)

        return similarities.tolist()

    def _extract_texts(
        self, put_ops: dict[tuple[tuple[str, ...], str], PutOp]
    ) -> dict[str, list[tuple[tuple[str, ...], str, str]]]:
        """Extract texts from put operations for embedding."""
        if put_ops and self.index_config and self.embeddings:
            to_embed = defaultdict(list)

            for op in put_ops.values():
                if op.value is not None and op.index is not False:
                    if op.index is None:
                        paths = self.index_config["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]

                    for path, field in paths:
                        texts = get_text_at_path(op.value, field)
                        if texts:
                            if len(texts) > 1:
                                for i, text in enumerate(texts):
                                    to_embed[text].append(
                                        (op.namespace, op.key, f"{path}.{i}")
                                    )
                            else:
                                to_embed[texts[0]].append((op.namespace, op.key, path))

            return to_embed

        return {}

    def _apply_put_ops(self, put_ops: dict[tuple[tuple[str, ...], str], PutOp]) -> None:
        """Apply put operations to MongoDB without embeddings."""
        now = datetime.now(timezone.utc)

        for (namespace, key), op in put_ops.items():
            doc_id = self._make_doc_id(namespace, key)

            if op.value is None:
                # Delete operation
                self._collection.delete_one({"_id": doc_id})
            else:
                # Upsert operation
                doc = {
                    "_id": doc_id,
                    "namespace": list(namespace),
                    "key": key,
                    "value": op.value,
                    "created_at": now,
                    "updated_at": now,
                }

                self._collection.replace_one(
                    {"_id": doc_id},
                    doc,
                    upsert=True,
                )

    def _apply_put_ops_with_embeddings(
        self,
        put_ops: dict[tuple[tuple[str, ...], str], PutOp],
        to_embed: dict[str, list[tuple[tuple[str, ...], str, str]]],
        embeddings: list[list[float]],
    ) -> None:
        """Apply put operations to MongoDB with embeddings."""
        now = datetime.now(timezone.utc)

        # Build embeddings map
        indices = [index for indices in to_embed.values() for index in indices]
        if len(indices) != len(embeddings):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) does not"
                f" match number of indices ({len(indices)})"
            )

        embedding_map: dict[tuple[tuple[str, ...], str], dict[str, list[float]]] = (
            defaultdict(dict)
        )
        for embedding, (ns, key, path) in zip(embeddings, indices, strict=False):
            embedding_map[(ns, key)][path] = embedding

        # Apply operations
        for (namespace, key), op in put_ops.items():
            doc_id = self._make_doc_id(namespace, key)

            if op.value is None:
                # Delete operation
                self._collection.delete_one({"_id": doc_id})
            else:
                # Upsert operation with embeddings
                doc = {
                    "_id": doc_id,
                    "namespace": list(namespace),
                    "key": key,
                    "value": op.value,
                    "created_at": now,
                    "updated_at": now,
                }

                # Add embeddings if available
                if (namespace, key) in embedding_map:
                    doc["embedding"] = embedding_map[(namespace, key)]

                self._collection.replace_one(
                    {"_id": doc_id},
                    doc,
                    upsert=True,
                )

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """List unique namespaces matching conditions."""
        # Get all unique namespaces
        pipeline = [
            {"$group": {"_id": "$namespace"}},
            {"$sort": {"_id": 1}},
        ]

        results = list(self._collection.aggregate(pipeline))
        namespaces = [tuple(doc["_id"]) for doc in results]

        # Apply match conditions
        if op.match_conditions:
            namespaces = [
                ns
                for ns in namespaces
                if all(self._does_match(condition, ns) for condition in op.match_conditions)
            ]

        # Apply max depth
        if op.max_depth is not None:
            namespaces = sorted({ns[: op.max_depth] for ns in namespaces})
        else:
            namespaces = sorted(namespaces)

        return namespaces[op.offset : op.offset + op.limit]

    def _does_match(self, condition: MatchCondition, namespace: tuple[str, ...]) -> bool:
        """Check if namespace matches a condition."""
        if condition.match_type == "prefix":
            prefix = tuple(condition.path.split("."))
            return namespace[: len(prefix)] == prefix
        elif condition.match_type == "suffix":
            suffix = tuple(condition.path.split("."))
            return namespace[-len(suffix) :] == suffix
        return False

    def __del__(self):
        """Clean up MongoDB connection."""
        if hasattr(self, "_client"):
            self._client.close()
