# PDF QA Builder 🧠

Read, chunk, summarize, embed, and query PDFs using Weaviate and LLM-based retrieval-augmented generation (RAG).

## Features

* Read one or more local or remote PDF files
* Detect sections (Abstract, Introduction, etc.) and split them into overlapping chunks
* Generate short summaries for each section or chunk
* Create embeddings and store them in a Weaviate vector database
* Retrieve semantically relevant chunks based on user questions
* Use an LLM to synthesize factual answers from retrieved context
* Flexible CLI for both indexing and querying

## Getting Started

### Prerequisites

* Python 3.11 or 3.12
* A local or containerized Weaviate instance (version ≥ 1.26)
* Optional: an OpenAI API key (for higher-quality summaries)

### Installation
```
git clone https://github.com/yourusername/pdf-qa-builder.git
cd pdf-qa-builder
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

Weaviate instance connection details are defined in the **docker-compose.ymal** file.

### Usage

#### 1. Build an index from PDFs

Read one or more PDFs, split them into chunks, embed, summarize, and store them in Weaviate.

```bash
python run.py index --pdf-dir data --chunk-size 250 --overlap 50
```

The directory data/ already includes five preloaded arXiv papers from October 2025, which can be used directly for indexing and question-answering.

#### 2. Ask questions

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Query the indexed PDFs and generate a synthesized answer from the LLM.

```bash
python run.py ask "What methods are used in this paper?"
```

## Architecture

1. **PDFReader** — Extracts text and detects structural sections (Abstract, Introduction, etc.)
2. **TextSplitter** — Splits sections into overlapping word chunks
3. **Summarizer** — Generates summaries (Hugging Face models or OpenAI)
4. **Embedder** — Creates embeddings for chunks and summaries
5. **VectorStore** — Connects to Weaviate, defines schema, stores and retrieves vectors
6. **Retriever** — Retrieves similar chunks based on a query
7. **Generator** — Builds prompts and uses an LLM to synthesize answers
8. **CLI (`run.py`)** — orchestrates `index`, and `ask` commands

## Commands / API

| Command | Description |
|---------|-------------|
| `run.py index --pdf-dir <path> --chunk-size <int> --overlap <int> --summary-level <section or chunk>` | Read PDFs, split into chunks, generate summaries, create embeddings, and upload to Weaviate |
| `run.py ask "question"` | Retrieve relevant chunks and generate an LLM answer |

#### Indexing parameters

* chunk-size	Number of words per chunk (default: 250)
* overlap	Overlapping words between chunks (default: 50)
* summary-level	Summarize at section or chunk level

## Examples

```
python run.py ask "In what ways do ACE and FlowSearch avoid losing information as their systems evolve?"


Connected to local Weaviate at http://localhost:8080
Embedding question: 'In what ways do ACE and FlowSearch avoid losing information as their systems evolve?'
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:19<00:00, 19.92s/it]
Querying Weaviate for top 5 matches...
Retrieved 5 results
Generating answer with gpt-4o-mini...

================================================================================
Question: In what ways do ACE and FlowSearch avoid losing information as their systems evolve?
================================================================================

ACE and FlowSearch employ several strategies to avoid losing information as their systems evolve:

1. **ACE (Agentic Context Engineering)**:
   - **Incremental Delta Updates**: ACE uses a system of incremental updates, allowing for localized edits rather than complete rewrites of the context. This means that only relevant parts of the context are updated, preserving existing knowledge while integrating new insights [Source 3].
   - **Structured Contexts**: ACE treats contexts as evolving playbooks that continuously accumulate and refine strategies over time. This structured approach helps maintain the quality and relevance of the context as it grows [Source 3].
   - **Reflector and Curator Roles**: The Reflector critiques and distills insights from the agent's experiences, while the Curator integrates these insights into the context. This separation of roles enhances the quality of the context and ensures that valuable information is retained and organized effectively [Source 3].
   - **Grow-and-Refine Mechanism**: This mechanism allows ACE to periodically refine the context, removing redundancy and ensuring that it remains compact and relevant, which helps in maintaining the integrity of the information [Source 3].

2. **FlowSearch**:
   - **Iterative Expansion**: FlowSearch employs an iterative process to expand its knowledge flow, gradually adding new nodes and dependencies. This method helps maintain structural coherence and logical consistency, ensuring that information is built upon rather than lost [Source 4].
   - **Knowledge Collector and Flow Refiner**: These components work together to identify executable nodes and refine the flow based on newly acquired knowledge. This iterative refinement process helps in retaining and enhancing the information gathered during execution [Source 4].

Together, these strategies enable ACE and FlowSearch to adapt and evolve while minimizing the risk of losing critical information.

================================================================================
Sources:
================================================================================
1. 2510.04618v1.pdf - Results
2. 2510.05592v1.pdf - SCALING TRENDS IN AGENTFLOW
3. 2510.04618v1.pdf - Agentic Context Engineering (ACE)
4. 2510.08521v1.pdf - 2.2. Knowledge Flow Planner
5. 2510.05592v1.pdf - MAIN RESULTS
```

**GPT5 Evaluation**

This is an excellent version of the question and answer — it’s balanced, technically correct, and readable.

|Criterion|Rating|Notes|
|---------|------|-------|
|Clarity|⭐⭐⭐⭐⭐ (5/5)	Nicely structured — clear sectioning, logical flow, consistent terminology.|
|Accuracy|⭐⭐⭐⭐☆ (4.5/5)	Correctly captures each framework’s main mechanism for preserving information.|
|Comparative focus|⭐⭐⭐⭐⭐ (5/5)	The answer draws parallel points (ACE’s deltas ↔ FlowSearch’s graph refinement).|
|Depth and completeness|⭐⭐⭐⭐☆ (4.5/5)	Covers the essential modules of both systems; could add one short sentence about FlowSearch’s DAG-based structure for extra precision.|
|Readability|⭐⭐⭐⭐⭐ (5/5)	Well formatted and approachable — ideal for both human readers and RAG evaluation.|

Final score: 4.8 / 5 — excellent.

---
```
python run.py ask "In what ways do Training-Free GRPO and AgentFlow balance optimization efficiency and generalization across reasoning tasks?"


Connected to local Weaviate at http://localhost:8080
Embedding question: 'In what ways do Training-Free GRPO and AgentFlow balance optimization efficiency and generalization across reasoning tasks?'
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:19<00:00, 19.92s/it]
Querying Weaviate for top 5 matches...
Retrieved 5 results
Generating answer with gpt-4o-mini...

================================================================================
Question: In what ways do Training-Free GRPO and AgentFlow balance optimization efficiency and generalization across reasoning tasks?
================================================================================

Training-Free GRPO and AgentFlow balance optimization efficiency and generalization across reasoning tasks in distinct yet complementary ways:

1. **Training-Free GRPO**:
   - **Generalization Across Domains**: Training-Free GRPO excels in cross-domain performance without the specialization trade-off seen in parameter-tuned models. It achieves strong results in diverse tasks by leveraging domain-specific learned experiences without fine-tuning, allowing it to maintain generalizability across different domains, such as mathematical reasoning and web searching tasks [Source 1].
   - **Cost Efficiency**: It significantly reduces training and inference costs compared to traditional reinforcement learning methods. For instance, it requires minimal training steps and resources, making it particularly advantageous in scenarios with limited data or budget constraints [Source 1].

2. **AgentFlow**:
   - **Enhanced Reasoning and Tool Utilization**: AgentFlow incorporates a dedicated agentic system that optimizes tool usage through in-the-flow reinforcement learning (Flow-GRPO). This allows it to adaptively select and utilize tools based on the task requirements, leading to improved performance across various reasoning tasks, including search-intensive and mathematical reasoning [Source 2].
   - **Robustness and Self-Correction**: The Flow-GRPO training enhances the agent's ability to self-correct and adapt its strategies in real-time, which is crucial for complex reasoning tasks. This adaptability enables AgentFlow to outperform larger monolithic models by effectively managing multi-turn reasoning and sub-goal decomposition [Source 2].

In summary, Training-Free GRPO emphasizes cost-effective generalization across diverse domains, while AgentFlow focuses on optimizing reasoning efficiency and tool utilization through adaptive learning, leading to superior performance in complex reasoning tasks.

================================================================================
Sources:
================================================================================
1. 2510.08191v1.pdf - Comparing RL Learning on Context Space and Parameter Space
2. 2510.05592v1.pdf - MAIN RESULT ANALYSIS
3. 2510.05592v1.pdf - TABLE OF CONTENTS
4. 2510.05592v1.pdf - CASE STUDIES
5. 2510.05592v1.pdf - MAIN RESULTS
```

**GPT5 Evaluation**

This is a strong, clear, and mostly accurate answer, but it slightly oversimplifies the mechanism that makes Training-Free GRPO generalize and the core optimization insight that differentiates it from AgentFlow. Let’s break it down carefully:

|Criterion|Rating|Notes|
|---------|------|-------|
|Accuracy|⭐⭐⭐⭐☆ (4.3/5)|The broad ideas are correct—Training-Free GRPO prioritizes cost-efficient generalization, while AgentFlow optimizes in-the-loop reasoning. Minor misphrasing around how GRPO achieves generalization.|
|Completeness|⭐⭐⭐⭐☆ (4/5)|Covers both methods’ goals and mechanisms; could add more about how GRPO works without gradient updates and why Flow-GRPO helps long-horizon credit assignment.|
|Clarity|⭐⭐⭐⭐⭐ (5/5)|Nicely structured, consistent phrasing, clear sub-sections.|
|Technical Depth|⭐⭐⭐⭐☆ (4/5)|Good, but could mention GRPO’s group-relative normalization or trajectory broadcasting in Flow-GRPO for precision.|

Overall	4.4 / 5 — Excellent, with minor technical refinements recommended.


