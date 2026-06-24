# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]>=0.8.4",
#     "shap>=0.46",
#     "scikit-learn>=1.5",
#     "xgboost>=2.0",
#     "skops>=0.11",
#     "numpy",
# ]
# ///
"""Stdio entry shim for the SHAP-explain VGI worker.

Lets the worker run straight from a source checkout (``uv run
explain_worker.py``) and from a container, and keeps ``import explain_worker``
working for tests. The implementation lives in ``vgi_explain.worker``; installed
users invoke the ``vgi-explain`` console script instead.

    ATTACH 'explain' (TYPE vgi, LOCATION 'uv run explain_worker.py');
"""

from vgi_explain.worker import ExplainWorker, main

__all__ = ["ExplainWorker", "main"]

if __name__ == "__main__":
    main()
