from typing import Union
from sac_rag.data_models import RetrievalMethod
from sac_rag.methods.baseline import BaselineRetrievalMethod, BaselineRetrievalStrategy
from sac_rag.methods.hybrid import HybridRetrievalMethod, HybridStrategy

AnyRetrievalConfig = Union[BaselineRetrievalStrategy, HybridStrategy]

def create_retriever(strategy_config: AnyRetrievalConfig, db_name: str | None = None) -> RetrievalMethod:
    if isinstance(strategy_config, BaselineRetrievalStrategy):
        print("Factory: Creating BaselineRetrievalMethod...")
        return BaselineRetrievalMethod(retrieval_strategy=strategy_config, db_name=db_name)
    elif isinstance(strategy_config, HybridStrategy):
        print("Factory: Creating HybridRetrievalMethod...")
        return HybridRetrievalMethod(retrieval_strategy=strategy_config)
    else:
        raise TypeError(f"Unhandled strategy configuration type: {type(strategy_config)}")
