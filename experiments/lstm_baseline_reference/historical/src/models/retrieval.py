"""Retrieval-augmented classifier.

Retrieves similar sketches from a reference collection drawn from the training
set and incorporates their representations into classification. The query
itself is excluded, and validation/test sketches are never in the collection.

TODO: Decide representation and similarity measure for retrieval.
TODO: Decide how retrieved examples are combined with the query.
TODO: Implement model.
"""
