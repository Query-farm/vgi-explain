# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]>=0.8.5",
#     "shap>=0.46",
#     "scikit-learn>=1.5",
#     "xgboost>=2.0",
#     "skops>=0.11",
#     "numpy",
# ]
# ///
"""HTTP entry shim for the SHAP-explain VGI worker (used by container deploys).

Forces the worker CLI into HTTP mode. The implementation lives in
``vgi_explain.worker``; installed users invoke the ``vgi-explain-http`` console
script instead.
"""

from vgi_explain.worker import ExplainWorker, main_http

__all__ = ["ExplainWorker", "main_http"]


def main() -> None:
    """Run the worker over HTTP."""
    main_http()


if __name__ == "__main__":
    main()
