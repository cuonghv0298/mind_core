from llama_index.core.query_engine import (
    CustomQueryEngine,
)
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    ImageNode,
    NodeWithScore,
    MetadataMode,
    TextNode,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.indices.query.schema import QueryBundle
from typing import Optional
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.colpali_rerank import ColPaliRerank

# colpali_reranker = ColPaliRerank(
#     top_n=3,
#     model="vidore/colpali-v1.2",
#     keep_retrieval_score=True,
#     device="cuda",  # or "cpu" or "cuda:0" or "mps" for Apple
# )
colpali_reranker = None


cohere_rerank = None

QA_PROMPT_TMPL = """\
Below we give parsed text and images as context.

Use both the parsed text and images to answer the question. 

---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query. Explain whether you got the answer
from the text or image, and if there's discrepancies, and your reasoning for the final answer.

Note:
- If you can't answer the question from the context, say that you don't know.
- Answer in the same language as the query.

Query: {query_str}
Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)


class MultimodalQueryEngine(CustomQueryEngine):
    """Custom multimodal Query Engine.

    Takes in a retriever to retrieve a set of document nodes.
    Also takes in a prompt template and multimodal model.

    """

    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: OpenAIMultiModal

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs) -> None:
        """Initialize."""
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def custom_query(self, query_str: str):
        # retrieve text nodes
        nodes = self.retriever.retrieve(query_str)
        image_nodes = [n for n in nodes if isinstance(n.node, ImageNode)]
        text_nodes = [n for n in nodes if isinstance(n.node, TextNode)]

        # Make QueryBundle
        query_bundle = QueryBundle(query_str)
        # reranking text nodes
        if cohere_rerank:
            text_nodes = cohere_rerank.postprocess_nodes(text_nodes, query_bundle)

        # reranking image nodes
        if colpali_reranker:
            image_nodes = colpali_reranker.postprocess_nodes(image_nodes, query_bundle)

        # create context string from text nodes, dump into the prompt
        context_str = "\n\n".join(
            [
                r.get_content(metadata_mode=MetadataMode.LLM) for r in text_nodes
            ]  # later can use reranked_text_nodes
        )
        fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)

        # synthesize an answer from formatted text and images
        llm_response_gen = self.multi_modal_llm.stream_complete(
            prompt=fmt_prompt,
            image_documents=[n.node for n in image_nodes],
        )

        def response_gen():
            for response in llm_response_gen:
                yield response.delta

        return StreamingResponse(
            response_gen=response_gen(),
            source_nodes=nodes,
            metadata={
                "text_nodes": text_nodes,
                "image_nodes": image_nodes,
            },
        )
