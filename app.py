import streamlit as st
import chromadb

from web_discovery import search_web_documents, download_pdf_from_url

from upload_processor import process_uploaded_pdf
from folder_ingestion import ingest_pdf_folder
from hybrid_search import search_and_generate_answer
from document_comparison import compare_documents
from research_report import generate_research_report


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial AI Research Assistant",
    page_icon="🏭",
    layout="wide"
)


# --------------------------------------------------
# CHROMADB DOCUMENT INFORMATION
# --------------------------------------------------

def get_document_library():

    try:

        chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        collection = chroma_client.get_collection(
            name="manufacturing_documents_local"
        )

        stored_data = collection.get(
            include=["metadatas"]
        )

        metadatas = stored_data.get(
            "metadatas",
            []
        )

        documents = {}

        for metadata in metadatas:

            if not metadata:
                continue

            source = metadata.get(
                "source"
            )

            if not source:
                continue

            category = metadata.get(
                "category",
                "Unknown"
            )

            page = metadata.get(
                "page"
            )

            if source not in documents:

                documents[source] = {
                    "name": source,
                    "category": category,
                    "pages": set(),
                    "chunks": 0
                }

            documents[source]["chunks"] += 1

            if page is not None:

                documents[source]["pages"].add(
                    page
                )

        library = []

        for document in documents.values():

            library.append(
                {
                    "name": document["name"],
                    "category": document["category"],
                    "pages": len(
                        document["pages"]
                    ),
                    "chunks": document["chunks"]
                }
            )

        library.sort(
            key=lambda x: x["name"].lower()
        )

        return library

    except Exception:

        return []


# --------------------------------------------------
# DELETE DOCUMENT
# --------------------------------------------------

def delete_document(document_name):

    try:

        chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        collection = chroma_client.get_collection(
            name="manufacturing_documents_local"
        )

        collection.delete(
            where={
                "source": document_name
            }
        )

        return True

    except Exception as error:

        print(
            f"Error deleting document: {error}"
        )

        return False


# --------------------------------------------------
# GET STORED DOCUMENT NAMES
# --------------------------------------------------

def get_stored_document_names():

    library = get_document_library()

    return [
        document["name"]
        for document in library
    ]


# --------------------------------------------------
# CLEAR RESEARCH
# --------------------------------------------------

def clear_research():

    st.session_state.research_result = None

    st.session_state.research_question = ""

    st.session_state.research_report = None

    st.session_state.research_input = ""


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "uploaded_document_names" not in st.session_state:

    st.session_state.uploaded_document_names = []


if "research_result" not in st.session_state:

    st.session_state.research_result = None


if "research_question" not in st.session_state:

    st.session_state.research_question = ""


if "research_report" not in st.session_state:

    st.session_state.research_report = None


if "research_input" not in st.session_state:

    st.session_state.research_input = ""


# Keep indexing results across Streamlit reruns so success messages
# remain visible after the document library refreshes.
if "last_upload_summary" not in st.session_state:
    st.session_state.last_upload_summary = None

if "last_folder_summary" not in st.session_state:
    st.session_state.last_folder_summary = None


# --------------------------------------------------
# LOAD DOCUMENTS FROM CHROMADB
# --------------------------------------------------

stored_documents = get_stored_document_names()

for document_name in stored_documents:

    if (
        document_name
        not in st.session_state.uploaded_document_names
    ):

        st.session_state.uploaded_document_names.append(
            document_name
        )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🏭 Industrial AI")

    st.markdown(
        "### Retrieval-Augmented Research Assistant"
    )

    st.divider()

    st.subheader(
        "📚 Document Library"
    )

    if st.session_state.uploaded_document_names:

        for document_name in (
            st.session_state.uploaded_document_names
        ):

            st.write(
                f"📄 {document_name}"
            )

    else:

        st.info(
            "No documents available."
        )

    st.divider()

    st.subheader(
        "⚙️ Features"
    )

    st.write(
        "🔎 Hybrid Search"
    )

    st.write(
        "📊 Document Comparison"
    )

    st.write(
        "🤖 AI Research Assistant"
    )

    st.write(
        "📑 Research Reports"
    )

    st.write(
        "📄 PDF Export"
    )


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title(
    "🏭 Industrial Knowledge Discovery Assistant"
)

st.write(
    "AI-powered research assistant for "
    "manufacturing and industrial documents."
)


# --------------------------------------------------
# KNOWLEDGE BASE DASHBOARD
# --------------------------------------------------

st.subheader(
    "📊 Knowledge Base Dashboard"
)

library = get_document_library()

document_count = len(
    library
)

chunk_count = sum(
    document["chunks"]
    for document in library
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Documents",
        document_count
    )


with col2:

    if document_count > 0:

        st.metric(
            "Knowledge Base",
            "Ready"
        )

    else:

        st.metric(
            "Knowledge Base",
            "Empty"
        )


with col3:

    if document_count > 0:

        st.metric(
            "Hybrid Search",
            "Available"
        )

    else:

        st.metric(
            "Hybrid Search",
            "Waiting"
        )


with col4:

    st.metric(
        "Stored Chunks",
        chunk_count
    )


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.divider()
st.subheader("📤 Upload Industrial Documents")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"{len(uploaded_files)} PDF(s) selected.")

    if st.button("Process Uploaded Documents"):
        processed_items = []
        errors = []
        progress = st.progress(0, text="Preparing to process PDFs...")

        for index, uploaded_file in enumerate(uploaded_files, start=1):
            progress.progress(
                (index - 1) / len(uploaded_files),
                text=f"Processing {uploaded_file.name} ({index}/{len(uploaded_files)})..."
            )
            try:
                result = process_uploaded_pdf(uploaded_file)
                processed_items.append(result)

                document_name = result["file_name"]
                if document_name not in st.session_state.uploaded_document_names:
                    st.session_state.uploaded_document_names.append(document_name)

            except Exception as error:
                errors.append({
                    "file": uploaded_file.name,
                    "error": str(error)
                })

            progress.progress(
                index / len(uploaded_files),
                text=f"Processed {index} of {len(uploaded_files)} PDF(s)"
            )

        st.session_state.last_upload_summary = {
            "processed": processed_items,
            "errors": errors,
            "selected": len(uploaded_files)
        }
        st.rerun()

# Persist the report after rerun so users can see the outcome.
if st.session_state.last_upload_summary:
    upload_summary = st.session_state.last_upload_summary
    st.subheader("📋 Latest Upload Result")

    if upload_summary["processed"]:
        st.success(
            f"Indexing complete: {len(upload_summary['processed'])} PDF(s) processed successfully."
        )
        for item in upload_summary["processed"]:
            st.write(
                f"✅ **{item['file_name']}** — "
                f"{item['pages']} pages, {item['chunks']} chunks, "
                f"category: {item['category']}"
            )

    if upload_summary["errors"]:
        st.error(f"{len(upload_summary['errors'])} PDF(s) failed to process.")
        for item in upload_summary["errors"]:
            st.write(f"❌ **{item['file']}** — {item['error']}")

    if not upload_summary["processed"] and not upload_summary["errors"]:
        st.info("No new upload results to display.")


# --------------------------------------------------

# BATCH FOLDER INGESTION (NO MANUAL FILE PICKING)
# --------------------------------------------------

st.divider()
st.subheader("🗂️ Batch PDF Folder Ingestion")
st.caption(
    "Place PDFs in a local folder and index them together. "
    "Subfolders are scanned recursively; already indexed files are skipped by default."
)

pdf_folder = st.text_input(
    "Local PDF folder path",
    value="./pdf_library",
    help="Example: ./pdf_library or an absolute path to your PDF repository."
)
force_reindex = st.checkbox(
    "Reprocess PDFs that are already indexed",
    value=False
)

if st.button("📚 Index PDF Folder"):
    try:
        with st.spinner("Scanning and indexing PDF folder..."):
            summary = ingest_pdf_folder(
                pdf_folder,
                categorize=False,
                force_reindex=force_reindex
            )

        # Save before rerunning. Otherwise st.rerun() clears the visible
        # one-time success messages before the user can read them.
        st.session_state.last_folder_summary = summary
        st.rerun()

    except Exception as error:
        st.session_state.last_folder_summary = {
            "folder": pdf_folder,
            "found": 0,
            "processed": [],
            "skipped": [],
            "errors": [{"file": pdf_folder, "error": str(error)}]
        }
        st.rerun()

# Display the latest folder result after any rerun, including the library refresh.
if st.session_state.last_folder_summary:
    summary = st.session_state.last_folder_summary
    st.subheader("📋 Latest Folder Indexing Result")
    st.write(f"**Folder:** `{summary.get('folder', pdf_folder)}`")
    st.write(f"**PDFs found:** {summary.get('found', 0)}")
    st.write(f"**PDFs processed:** {len(summary.get('processed', []))}")
    st.write(f"**Already indexed / skipped:** {len(summary.get('skipped', []))}")
    st.write(f"**Errors:** {len(summary.get('errors', []))}")

    if summary.get("processed"):
        st.success(
            f"Indexing complete: {len(summary['processed'])} new PDF(s) indexed successfully."
        )
        for item in summary["processed"]:
            st.write(
                f"✅ **{item['file_name']}** — "
                f"{item['pages']} pages, {item['chunks']} chunks, "
                f"category: {item['category']}"
            )

    if summary.get("skipped"):
        for item in summary["skipped"]:
            st.info(f"⏭️ Skipped already indexed: {item}")

    if summary.get("errors"):
        st.error(f"{len(summary['errors'])} PDF(s) failed.")
        for item in summary["errors"]:
            st.error(f"❌ {item.get('file', 'Unknown file')} — {item.get('error', 'Unknown error')}")

    if not summary.get("processed") and not summary.get("errors"):
        st.info("Folder scan complete. No new PDFs needed indexing.")


# --------------------------------------------------
# WEB DOCUMENT DISCOVERY
# --------------------------------------------------

st.divider()
st.subheader("🌐 Discover Industrial Documents on the Web")
st.caption(
    "Search for relevant technical documents. Review the results first; only direct PDF links can be imported here. "
    "Web results are not added to ChromaDB unless you explicitly import them."
)

if "web_discovery_results" not in st.session_state:
    st.session_state.web_discovery_results = []

with st.form("web_document_search_form"):
    web_query = st.text_input(
        "What industrial topic are you researching?",
        placeholder="e.g. additive manufacturing process parameters, CNC maintenance, mechanical design",
    )
    web_result_limit = st.slider("Maximum search results", min_value=3, max_value=10, value=8)
    run_web_search = st.form_submit_button("🔎 Search the Web")

if run_web_search:
    try:
        with st.spinner("Searching the web for relevant industrial resources..."):
            st.session_state.web_discovery_results = search_web_documents(web_query, web_result_limit)
    except Exception as error:
        st.error(f"Web search failed: {error}")
        st.session_state.web_discovery_results = []

if st.session_state.web_discovery_results:
    st.write(f"Found {len(st.session_state.web_discovery_results)} result(s). Select direct PDF links you want to import.")
    selected_web_results = []
    for result_index, result in enumerate(st.session_state.web_discovery_results):
        with st.container(border=True):
            st.markdown(f"**{result['title']}**")
            st.caption(result['url'])
            if result.get("snippet"):
                st.write(result["snippet"][:700])
            if result.get("score") is not None:
                st.caption(f"Search relevance score: {float(result['score']):.2f}")
            if result.get("is_pdf_url"):
                if st.checkbox("Select this PDF for import", key=f"web_pdf_select_{result_index}"):
                    selected_web_results.append(result)
            else:
                st.info("This result appears to be a webpage, not a direct PDF. Open it to inspect; this importer currently accepts direct PDF URLs.")
            st.markdown(f"[Open source ↗]({result['url']})")

    if selected_web_results and st.button("📥 Import Selected PDFs into Existing RAG Index"):
        imported = []
        import_errors = []
        progress = st.progress(0, text="Preparing selected web PDFs...")
        for idx, result in enumerate(selected_web_results, start=1):
            try:
                pdf_stream = download_pdf_from_url(result["url"])
                # Prefix with web_ to reduce filename collisions with local PDFs.
                pdf_stream.name = "web_" + pdf_stream.name
                outcome = process_uploaded_pdf(pdf_stream)
                imported.append({"title": result["title"], **outcome, "source_url": result["url"]})
                if outcome.get("file_name") not in st.session_state.uploaded_document_names:
                    st.session_state.uploaded_document_names.append(outcome.get("file_name"))
            except Exception as error:
                import_errors.append({"url": result["url"], "error": str(error)})
            progress.progress(idx / len(selected_web_results), text=f"Processed {idx} of {len(selected_web_results)} selected PDF(s)")

        st.session_state.last_web_import_summary = {"processed": imported, "errors": import_errors}
        st.rerun()

if st.session_state.get("last_web_import_summary"):
    web_summary = st.session_state.last_web_import_summary
    st.subheader("📋 Latest Web Import Result")
    if web_summary.get("processed"):
        st.success(f"Imported {len(web_summary['processed'])} web PDF(s) into the existing RAG index.")
        for item in web_summary["processed"]:
            st.write(f"✅ **{item.get('file_name', 'PDF')}** — {item.get('pages', 0)} pages, {item.get('chunks', 0)} chunks")
            st.caption(f"Source: {item.get('source_url', '')}")
    if web_summary.get("errors"):
        st.warning(f"{len(web_summary['errors'])} PDF(s) could not be imported.")
        for item in web_summary["errors"]:
            st.write(f"❌ {item['url']} — {item['error']}")


# --------------------------------------------------

# DOCUMENT LIBRARY
# --------------------------------------------------

st.divider()

st.subheader(
    "📚 Document Library"
)

library = get_document_library()

if library:

    for document in library:

        with st.container(
            border=True
        ):

            col1, col2, col3, col4, col5 = st.columns(
                [3, 2, 1, 1, 1]
            )

            with col1:

                st.write(
                    f"📄 **{document['name']}**"
                )

            with col2:

                st.write(
                    f"🏷️ {document['category']}"
                )

            with col3:

                st.write(
                    f"📑 {document['pages']} pages"
                )

            with col4:

                st.write(
                    f"🧩 {document['chunks']} chunks"
                )

            with col5:

                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{document['name']}"
                ):

                    deleted = delete_document(
                        document["name"]
                    )

                    if deleted:

                        if (
                            document["name"]
                            in st.session_state.uploaded_document_names
                        ):

                            st.session_state.uploaded_document_names.remove(
                                document["name"]
                            )

                        st.toast(
                            f"Deleted {document['name']}",
                            icon="✅"
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Could not delete the document."
                        )

else:

    st.info(
        "No documents stored in the knowledge base."
    )


# --------------------------------------------------
# DOCUMENT COMPARISON
# --------------------------------------------------

st.divider()

st.subheader(
    "📊 Multi-Document Comparison"
)

document_names = [
    document["name"]
    for document in library
]

if len(document_names) >= 2:

    selected_documents = st.multiselect(
        "Select documents to compare",
        document_names
    )

    if st.button(
        "Compare Selected Documents"
    ):

        if len(selected_documents) < 2:

            st.warning(
                "Please select at least two documents."
            )

        else:

            with st.spinner(
                "Comparing documents..."
            ):

                comparison_result = compare_documents(
                    selected_documents
                )

            st.subheader(
                "Comparison Result"
            )

            st.markdown(
                comparison_result["answer"]
            )

else:

    st.info(
        "Upload at least two documents "
        "to enable document comparison."
    )


# --------------------------------------------------
# RESEARCH ASSISTANT
# --------------------------------------------------

st.divider()

st.subheader(
    "🤖 Industrial Research Assistant"
)

question = st.text_area(
    "Enter your research question",
    placeholder=(
        "Example: What are the different "
        "types of manufacturing processes?"
    ),
    height=100,
    key="research_input"
)


# --------------------------------------------------
# RESEARCH BUTTONS
# --------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    generate_clicked = st.button(
        "🔎 Generate Answer"
    )


with col2:

    st.button(
        "🗑️ Clear Research",
        on_click=clear_research
    )


# --------------------------------------------------
# GENERATE RESEARCH ANSWER
# --------------------------------------------------

if generate_clicked:

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

    else:

        with st.spinner(
            "Searching industrial knowledge base..."
        ):

            result = search_and_generate_answer(
                question
            )

        st.session_state.research_result = result

        st.session_state.research_question = (
            question
        )

        st.session_state.research_report = None


# --------------------------------------------------
# DISPLAY RESEARCH RESULT
# --------------------------------------------------

if st.session_state.research_result:

    result = st.session_state.research_result

    st.subheader(
        "💡 Research Answer"
    )

    st.markdown(
        result["answer"]
    )

    st.divider()

    confidence = result.get(
        "confidence",
        0
    )

    confidence_label = result.get(
        "confidence_label",
        "Low"
    )

    st.subheader(
        "🎯 Retrieval Confidence"
    )

    st.progress(
        confidence
    )

    st.write(
        f"{confidence * 100:.1f}% — "
        f"{confidence_label} confidence"
    )


    # --------------------------------------------------
    # RESEARCH REPORT
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "📑 Research Report"
    )

    if st.button(
        "Generate Research Report"
    ):

        with st.spinner(
            "Generating research report..."
        ):

            report = generate_research_report(
                st.session_state.research_question,
                result["answer"],
                result["results"],
                confidence,
                confidence_label
            )

        st.session_state.research_report = (
            report
        )


    if st.session_state.research_report:

        st.markdown(
            st.session_state.research_report
        )


        # --------------------------------------------------
        # PDF EXPORT
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "📄 Export Report"
        )

        from reportlab.lib.pagesizes import A4

        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer
        )

        from reportlab.lib.styles import (
            getSampleStyleSheet
        )

        from io import BytesIO

        pdf_buffer = BytesIO()

        pdf_document = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4
        )

        styles = getSampleStyleSheet()

        story = []

        for line in (
            st.session_state.research_report
            .split("\n")
        ):

            if line.strip():

                story.append(
                    Paragraph(
                        line.replace(
                            "&",
                            "&amp;"
                        ),
                        styles["BodyText"]
                    )
                )

                story.append(
                    Spacer(
                        1,
                        6
                    )
                )

        pdf_document.build(
            story
        )

        st.download_button(
            label=(
                "⬇️ Download Research Report PDF"
            ),
            data=pdf_buffer.getvalue(),
            file_name=(
                "industrial_research_report.pdf"
            ),
            mime="application/pdf"
        )


# --------------------------------------------------
# RETRIEVED SOURCES
# --------------------------------------------------

if st.session_state.research_result:

    discovered_documents = st.session_state.research_result.get(
        "discovered_documents", []
    )
    discovery_enabled = st.session_state.research_result.get(
        "document_discovery_enabled", False
    )

    st.divider()
    st.subheader("🧭 Discovered Relevant Documents")

    if discovery_enabled and discovered_documents:
        st.caption(
            "These PDFs were ranked at document level before chunk retrieval."
        )
        for index, document in enumerate(discovered_documents, start=1):
            with st.expander(
                f"{index}. {document['source']} — {document.get('category', 'Unknown')}"
            ):
                st.write(f"**Pages:** {document.get('pages', 'Unknown')}")
                st.write(f"**Indexed chunks:** {document.get('chunks', 'Unknown')}")
                if document.get("distance") is not None:
                    st.write(f"**Vector distance:** {document['distance']:.4f}")
                st.write(document.get("catalog_preview", ""))
    else:
        st.info(
            "Document-level discovery is not active yet. "
            "Run `python build_document_catalog.py` once to create the catalogue "
            "for PDFs already in ChromaDB. Newly processed PDFs will be added automatically."
        )

    st.divider()

    st.subheader(
        "📚 Retrieved Sources"
    )

    results = (
        st.session_state.research_result[
            "results"
        ]
    )

    for index, (
        chunk_id,
        item
    ) in enumerate(
        results,
        start=1
    ):

        metadata = item["metadata"]

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        category = metadata.get(
            "category",
            "Unknown"
        )

        score = item.get(
            "hybrid_score",
            0
        )

        with st.expander(
            f"Source {index}: {source} — Page {page}"
        ):

            st.write(
                f"**Category:** {category}"
            )

            st.write(
                f"**Retrieval Score:** {score:.4f}"
            )

            st.write(
                item.get(
                    "document",
                    ""
                )
            )