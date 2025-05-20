from typing import List


def evaluate_multimodal_rag(
    input_path: Path,
    n_questions: int,
    override: bool,
    image_k: int,
    retrievers: List[RetrieverRecord] = None,
):
    # Import dependencies
    from pathlib import Path
    import glob
    import os
    import qdrant_client
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.core import StorageContext
    from llama_index.core.indices import MultiModalVectorStoreIndex
    import shutil
    import pandas as pd
    from utils.loaders import parse_pdf_2_png
    from pydantic import BaseModel, Field
    from llama_index.multi_modal_llms.gemini import GeminiMultiModal
    from llama_index.core.program import MultiModalLLMCompletionProgram
    from llama_index.core.output_parsers import PydanticOutputParser
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.schema import NodeWithScore
    from llama_index.core.base.base_retriever import BaseRetriever

    # get the pdf name
    pdf_name = input_path.stem
    parent = input_path.parent
    image_output_dir = Path(f"{input_path.parent}/{pdf_name}_images")
    image_question_answer_dir = Path(f"{input_path.parent}/{pdf_name}_eval_data")
    gemini_llm = GeminiMultiModal(model_name="models/gemini-2.0-flash")

    # Models
    class Pairs(BaseModel):
        question: str = Field(
            default="", description="Question generated from specific image."
        )
        answer: str = Field(default="", description="Answer to the question")

    class ImageWithAnswer(BaseModel):
        image_path: str = Field(
            default="",
            description="Path to image which is stored in metadata of each ImageDocument",
        )
        data: List[Pairs] = Field(
            default_factory=list, description="List of pairs of question and answer"
        )

    class MetadataAwareDataGenerator:
        def __init__(self):
            pass

        def generate(self, image_documents) -> List[ImageWithAnswer]:
            results = []
            for img_doc in image_documents:
                file_path = img_doc.metadata["file_name"]
                QUESTION_GENERATOR_PROMPT = f"""
Based on the given image located at {file_path}, generate {n_questions} pairs of question and answer in the following JSON format.
Note that image_path in json response should contain field file_path in metadata of each ImageDocument.
"""
                llm_program = MultiModalLLMCompletionProgram.from_defaults(
                    output_parser=PydanticOutputParser(ImageWithAnswer),
                    image_documents=[img_doc],
                    prompt_template_str=QUESTION_GENERATOR_PROMPT,
                    multi_modal_llm=gemini_llm,
                    verbose=True,
                )
                response = llm_program()
                results.append(response)
            return results

    # Step 1: Convert pdf to images
    if not os.path.isdir(image_output_dir) or override:
        print("Output directory does not exist. Creating...")
        parse_pdf_2_png(input_file_path=input_path, output_dir=image_output_dir)

    img_paths = glob.glob(os.path.join(image_output_dir, "*.png"))
    # Load the image document
    image_documents = SimpleDirectoryReader(input_files=img_paths).load_data()

    # Generate the image Q&A pairs
    if not os.path.isdir(image_question_answer_dir) or override:
        data_generator = MetadataAwareDataGenerator()
        results = data_generator.generate(image_documents)
        results = [result.model_dump() for result in results]
        rs_df = pd.DataFrame(results)
        image_question_answer_dir.mkdir(parents=True, exist_ok=True)
        rs_df.to_parquet(f"{image_question_answer_dir}/image_qa.parquet")
    else:
        results = pd.read_parquet(f"{image_question_answer_dir}/image_qa.parquet")
        results = results.to_dict("records")

    # Evaluation metrics
    def mrr_single(answer, facts):
        facts = [facts] if not isinstance(facts, list) else facts
        for rank, pred in enumerate(answer, start=1):
            if pred in facts:
                return 1 / rank
        return 0.0

    def precision_at_k(answer, facts, k=None):
        facts = [facts] if not isinstance(facts, list) else facts
        if k is None:
            k = len(answer)
        answer_k = answer[:k]
        relevant = sum(1 for item in answer_k if item in facts)
        return relevant / len(answer_k) if answer_k else 0.0

    def recall_at_k(answer, facts, k=None):
        facts = [facts] if not isinstance(facts, list) else facts
        if k is None:
            k = len(answer)
        answer_k = answer[:k]
        relevant = sum(1 for item in answer_k if item in facts)
        return relevant / len(facts) if facts else 0.0

    def map_llm_retrieval_result_to_evaluation_format(
        retrieved_nodes: List[NodeWithScore],
    ):
        return [n.node.metadata["file_name"] for n in retrieved_nodes]

    # Create baseline retriever
    vectordb_folder_path = "qdrant_db"
    if os.path.exists(vectordb_folder_path):
        shutil.rmtree(vectordb_folder_path)
    client = qdrant_client.QdrantClient(path=vectordb_folder_path)

    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )
