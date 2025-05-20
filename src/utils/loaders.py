from typing import List
from pathlib import Path

import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema.document import Document
from langchain_community.document_loaders import WebBaseLoader


def extract_pages(file) -> List[Document]:
    """Extracts the pages from a PDF document"""
    reader = PdfReader(file)
    number_of_pages = len(reader.pages)

    pages = []

    for i in range(number_of_pages):
        page = reader.pages[i]
        text = page.extract_text()
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": (
                        file.name
                        if type(file) == st.runtime.uploaded_file_manager.UploadedFile
                        else ""
                    ),
                    "page": i,
                },
            )
        )

    return pages


@st.cache_data
def extract_web_content(url: str) -> List[Document]:
    """Extracts content from a web page"""
    loader = WebBaseLoader(url)
    docs = loader.load()

    return docs


def parse_pdf_2_png(
    input_file_path: Path, output_dir: Path, image_resolution_scale: float = 2.0
):
    import os
    from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
    import time
    from pathlib import Path
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Using Docling

    pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_cell_matching = False
    
    pipeline_options.images_scale = image_resolution_scale
    pipeline_options.generate_page_images = True
    

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_res = doc_converter.convert(input_file_path)

    doc_filename = conv_res.input.file.stem

    # Save page images
    for page_no, page in conv_res.document.pages.items():
        page_no = page.page_no
        page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # Save images of figures and tables
    # table_counter = 0
    # picture_counter = 0
    # for element, _level in conv_res.document.iterate_items():
    #     if isinstance(element, TableItem):
    #         table_counter += 1
    #         element_image_filename = (
    #             output_dir / f"{doc_filename}-table-{table_counter}.png"
    #         )
    #         with element_image_filename.open("wb") as fp:
    #             element.get_image(conv_res.document).save(fp, "PNG")
    #     if isinstance(element, PictureItem):
    #         picture_counter += 1
    #         element_image_filename = (
    #             output_dir / f"{doc_filename}-picture-{picture_counter}.png"
    #         )
    #         with element_image_filename.open("wb") as fp:
    #             element.get_image(conv_res.document).save(fp, "PNG")

    # Export Markdown format:
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_res.document.export_to_markdown())

    # Using pdf2image
    # from pdf2image import convert_from_path
    # images = convert_from_path(input_file_path, dpi=300 * image_resolution_scale)
    # for i, image in enumerate(images):
    #     image.save(output_dir / f"{input_file_path.stem}-{i}.png", "PNG")

    end_time = time.time() - start_time

    print(f"Document converted and figures exported in {end_time:.2f} seconds.")


def load_documents(input_dir: Path):
    from llama_index.core import SimpleDirectoryReader
    from llama_index.readers.file import ImageReader

    output_dir = Path(f"{input_dir.parent}/{input_dir.stem}_preprocessed_data")
    parse_pdf_2_png(input_dir, output_dir)
    image_reader = ImageReader(keep_image=True, parse_text=False)

    # Define a custom file extractor for image files
    file_extractor = {
        ".jpg": image_reader,
        ".png": image_reader,
        ".jpeg": image_reader,
    }
    documents = SimpleDirectoryReader(
        input_dir=output_dir,
        recursive=True,
        file_extractor=file_extractor,
    ).load_data()
    return documents
