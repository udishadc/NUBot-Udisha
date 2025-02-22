import json
import time
import logging
import os
from typing import List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
import numpy as np
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = ""

os.environ['HF_TOKEN'] = ''


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Unified configuration settings for the evaluation system."""
    
    # Directory settings
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = BASE_DIR / "evaluation_results"
    PERSIST_DIR = BASE_DIR / "vector_db"

    
    @classmethod
    def get_persist_dir(cls, dimension: int) -> Path:
        """Get dimension-specific persistence directory."""
        persist_dir = cls.BASE_DIR / "vector_db" / f"dim_{dimension}"
        persist_dir.mkdir(parents=True, exist_ok=True)
        return persist_dir

    # Chunk settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 3

    # Model settings
    LLAMA_MODELS = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-1B"
    ]
    
    GPT_MODELS = [
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4"
    ]
    
    EMBEDDING_MODELS = {
        "llama": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ],
        "gpt": [
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
    }

    # Evaluation thresholds
    SIMILARITY_THRESHOLD = 0.8
    PARTIAL_THRESHOLD = 0.5
    EMBEDDING_DIMENSIONS = {
        # OpenAI embeddings
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        # HuggingFace embeddings
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768
    }

    @classmethod
    def setup(cls):
        """Create necessary directories."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.PERSIST_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for a model combination."""
    model_type: str  # 'llama' or 'gpt'
    model_name: str
    embedding_name: str
    description: str = ""


class UnifiedRAGSystem:
    """Implementation of the unified RAG system supporting both LLaMA and GPT models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the RAG system with either LLaMA or GPT model."""
        self.config = config
        self.model_type = config.model_type
        logger.info(f"Initializing {self.model_type.upper()} RAG system with model: {config.model_name}")
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful academic assistant. Provide clear, concise, and accurate responses based solely on the provided context.

Context:
{context}

Question: {question}

Response:"""
        )
        
        if self.model_type == "llama":
            self._init_llama_model()
        else:
            self._init_gpt_model()

        self.embeddings = None
        self.vectorstore = None
        self.qa_system = None

    def _init_llama_model(self):
        """Initialize LLaMA model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            truncation=True,
            return_full_text=False
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def _init_gpt_model(self):
        """Initialize GPT model."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
            
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=0.3,
            max_tokens=256
        )

    def set_embeddings(self):
        """Set the embedding model based on model type."""
        if self.model_type == "llama":
            self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_name)
        else:
            self.embeddings = OpenAIEmbeddings(model=self.config.embedding_name)

    def process_documents(self, jsonl_files: List[Path]) -> List[Dict[str, Any]]:
        """Process JSONL files into formatted documents."""
        documents = []
        for file in jsonl_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        doc = self._format_document(data, str(file))
                        documents.append(doc)
                    except Exception as e:
                        logger.error(f"Error processing {file}: {str(e)}")
        return documents

    def _format_document(self, data: Dict, source_file: str) -> Dict[str, Any]:
        """Format a document with metadata."""
        keywords = ", ".join(data['summary']['metadata']['keywords'])
        
        text = f"""
Title: {Path(source_file).stem}
Summary: {data['summary']['summary']}
Key Points: {' • '.join(data['summary']['key_points'])}
Type: {data['summary']['document_type']}
Audience: {data['summary']['target_audience']}
Keywords: {keywords}
Action Items: {' • '.join(f"{item['action']} (Due: {item['deadline']})"
                       for item in data['summary']['action_items'])}
"""
        
        return {
            "text": text.strip(),
            "metadata": {
                "source": source_file,
                "type": data['summary']['document_type'],
                "keywords": keywords
            }
        }

    def initialize(self, root_dir: str) -> None:
        """Initialize the system with dimension-specific database."""
        if not self.embeddings:
            self.set_embeddings()
            
        dimension = Config.EMBEDDING_DIMENSIONS[self.config.embedding_name]
        persist_dir = Config.get_persist_dir(dimension)
        
        root_path = Path(root_dir)
        jsonl_files = list(root_path.rglob("*.jsonl"))
        
        documents = self.process_documents(jsonl_files)
        chunks = self._create_chunks(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(persist_dir)
        )
        
        self.initialize_qa_system()

    def _create_chunks(self, documents: List[Dict[str, Any]]) -> List[Any]:
        """Create document chunks for processing."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            doc_chunks = splitter.create_documents(
                texts=[doc["text"]],
                metadatas=[doc["metadata"]]
            )
            chunks.extend(doc_chunks)
        return chunks

    def initialize_qa_system(self) -> None:
        """Initialize the QA system with existing vectorstore."""
        if not self.vectorstore:
            raise ValueError("Vectorstore must be initialized first")

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": Config.TOP_K_RETRIEVAL,
                "lambda_mult": 0.7
            }
        )

        self.qa_system = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": self.prompt_template,
                "verbose": True
            }
        )

    def query(self, question: str) -> str:
        """Process a question and return response."""
        if not self.qa_system:
            raise ValueError("System not initialized")

        try:
            response = self.qa_system.run(question.strip())
            return response.strip()
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return "Error processing question"

class UnifiedModelEvaluator:
    """Evaluator for multiple model configurations across different model types."""
    
    def __init__(self, test_cases: List[Dict], model_configs: List[ModelConfig]):
        """Initialize the evaluator."""
        self.test_cases = test_cases
        self.model_configs = model_configs
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    def evaluate_all_models(self, root_directory: str) -> Dict[str, Any]:
        """Evaluate models with dimension-specific databases."""
        results = {
            "model_comparisons": [],
            "detailed_results": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dimension_groups": {}
        }

        dimension_groups = {}
        for config in self.model_configs:
            dimension = Config.EMBEDDING_DIMENSIONS[config.embedding_name]
            if dimension not in dimension_groups:
                dimension_groups[dimension] = []
            dimension_groups[dimension].append(config)

        for dimension, configs in dimension_groups.items():
            logger.info(f"Processing dimension group: {dimension}")
            persist_dir = Config.get_persist_dir(dimension)
            
            vectorstore = None
            for config in configs:
                model_id = f"{config.model_type}_{config.model_name}_{config.embedding_name}"
                logger.info(f"Evaluating: {model_id}")

                try:
                    rag_system = UnifiedRAGSystem(config)
                    if vectorstore is None:
                        rag_system.initialize(root_directory)
                        vectorstore = rag_system.vectorstore
                    else:
                        rag_system.vectorstore = vectorstore
                        rag_system.initialize_qa_system()

                    model_results = self._evaluate_model(rag_system)
                    
                    results["detailed_results"][model_id] = {
                        "config": config.__dict__,
                        "results": model_results,
                        "embedding_dimension": dimension,
                        "database_path": str(persist_dir)
                    }
                    
                    results["model_comparisons"].append({
                        "model_id": model_id,
                        "model_type": config.model_type,
                        "embedding_dimension": dimension,
                        "average_time": model_results["average_time"],
                        "average_similarity": model_results["average_similarity"]
                    })

                except Exception as e:
                    logger.error(f"Error evaluating {model_id}: {str(e)}")
                    continue

            results["dimension_groups"][dimension] = str(persist_dir)

        return results

    def _evaluate_model(self, rag_system: UnifiedRAGSystem) -> Dict[str, Any]:
        """Evaluate a single model configuration."""
        results = {
            "questions": [],
            "average_time": 0,
            "average_similarity": 0
        }
    
        total_time = 0
        total_similarity = 0
    
        for test_case in self.test_cases:
            start_time = time.time()
            response = rag_system.query(test_case["question"])
            response_time = time.time() - start_time
    
            rouge_scores = self.scorer.score(test_case["answer"], response)
            similarity = rouge_scores['rougeL'].fmeasure
    
            results["questions"].append({
                "question": test_case["question"],
                "expected": test_case["answer"],
                "actual": response,
                "similarity": similarity,
                "response_time": response_time
            })
    
            total_time += response_time
            total_similarity += similarity
    
        results["average_time"] = total_time / len(self.test_cases)
        results["average_similarity"] = total_similarity / len(self.test_cases)
    
        return results

    def create_evaluation_dataframe(self, results: Dict) -> tuple:
        """Create comprehensive evaluation DataFrames."""
        records = []
        
        for model_id, details in results["detailed_results"].items():
            model_type = details["config"]["model_type"]
            model_name = details["config"]["model_name"]
            embedding_name = details["config"]["embedding_name"]
            
            for question_result in details["results"]["questions"]:
                records.append({
                    "model_id": model_id,
                    "model_type": model_type,
                    "model_name": model_name,
                    "embedding_model": embedding_name,
                    "question": question_result["question"],
                    "expected_answer": question_result["expected"],
                    "actual_answer": question_result["actual"],
                    "similarity_score": question_result["similarity"],
                    "response_time": question_result["response_time"]
                })
        
        df = pd.DataFrame(records)
        
        # Calculate summary statistics
        summary_stats = df.groupby(["model_type", "model_id"]).agg({
            "similarity_score": ["mean", "std", "min", "max"],
            "response_time": ["mean", "std", "min", "max"]
        }).round(4)
        
        # Add analysis columns
        df["answer_length_diff"] = df.apply(
            lambda x: len(x["actual_answer"]) - len(x["expected_answer"]), axis=1
        )
        
        df["response_category"] = df["similarity_score"].apply(
            lambda x: "Excellent" if x >= 0.8 else 
                     "Good" if x >= 0.6 else 
                     "Fair" if x >= 0.4 else 
                     "Poor"
        )
        
        return df, summary_stats

    def save_results(self, results: Dict, output_dir: str = "evaluation_results"):
        """Save evaluation results and generate visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create and save DataFrames
        detailed_df, summary_df = self.create_evaluation_dataframe(results)
        
        # Save detailed results
        detailed_df.to_csv(output_path / "detailed_results.csv", index=False, encoding='utf-8')
        
        # Save summary statistics
        summary_df.to_csv(output_path / "model_statistics.csv")
        
        # Save results in different formats
        self._save_model_type_comparisons(detailed_df, output_path)
        self._create_visualizations(results, output_path, detailed_df)
        self._generate_comprehensive_report(results, detailed_df, summary_df, output_path)

    def _save_model_type_comparisons(self, df: pd.DataFrame, output_path: Path):
        """Save separate comparison files for different aspects of the evaluation."""
        # Performance by model type
        model_type_stats = df.groupby("model_type").agg({
            "similarity_score": ["mean", "std", "min", "max"],
            "response_time": ["mean", "std", "min", "max"]
        }).round(4)
        model_type_stats.to_csv(output_path / "model_type_comparison.csv")

        # Detailed performance comparison
        performance_comparison = df.groupby(["model_type", "model_name", "embedding_model"]).agg({
            "similarity_score": ["mean", "std"],
            "response_time": ["mean", "std"],
            "response_category": lambda x: x.value_counts().index[0]  # Most common category
        }).round(4)
        performance_comparison.to_csv(output_path / "detailed_performance_comparison.csv")

        # Question-wise comparison
        question_comparison = df.pivot_table(
            index="question",
            columns=["model_type", "model_name"],
            values=["similarity_score", "response_time"],
            aggfunc=["mean", "std"]
        ).round(4)
        question_comparison.to_csv(output_path / "question_wise_comparison.csv")

    def _create_visualizations(self, results: Dict, output_path: Path, df: pd.DataFrame):
        """Create comprehensive visualizations comparing all models."""        
        # 1. Overall Performance Comparison
        self._create_overall_performance_plot(df, output_path)
        
        # 2. Model Type Comparison
        self._create_model_type_comparison_plot(df, output_path)
        
        # 3. Response Time Distribution
        self._create_response_time_plot(df, output_path)
        
        # 4. Response Quality Distribution
        self._create_quality_distribution_plot(df, output_path)
        
        # 5. Performance by Question Type
        self._create_question_performance_plot(df, output_path)

    def _create_overall_performance_plot(self, df: pd.DataFrame, output_path: Path):
        """Create overall performance comparison plot."""
        plt.figure(figsize=(15, 8))
        
        # Create scatter plot with different markers for different model types
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            plt.scatter(
                model_data['response_time'],
                model_data['similarity_score'],
                label=model_type.upper(),
                alpha=0.6,
                s=100
            )

        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Similarity Score')
        plt.title('Model Performance Comparison: Response Time vs Similarity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / 'overall_performance.png')
        plt.close()

    def _create_model_type_comparison_plot(self, df: pd.DataFrame, output_path: Path):
        """Create model type comparison plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Similarity comparison
        sns.boxplot(data=df, x='model_type', y='similarity_score', ax=ax1)
        ax1.set_title('Similarity Scores by Model Type')
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Similarity Score')

        # Response time comparison
        sns.boxplot(data=df, x='model_type', y='response_time', ax=ax2)
        ax2.set_title('Response Times by Model Type')
        ax2.set_xlabel('Model Type')
        ax2.set_ylabel('Response Time (seconds)')

        plt.tight_layout()
        plt.savefig(output_path / 'model_type_comparison.png')
        plt.close()

    def _create_response_time_plot(self, df: pd.DataFrame, output_path: Path):
        """Create response time distribution plot."""
        plt.figure(figsize=(12, 6))
        
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            sns.kdeplot(
                data=model_data['response_time'],
                label=model_type.upper(),
                fill=True,
                alpha=0.3
            )

        plt.title('Response Time Distribution by Model Type')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'response_time_distribution.png')
        plt.close()

    def _create_quality_distribution_plot(self, df: pd.DataFrame, output_path: Path):
        """Create response quality distribution plot."""
        plt.figure(figsize=(12, 6))
        
        quality_data = pd.crosstab(df['model_type'], df['response_category'])
        quality_data_pct = quality_data.div(quality_data.sum(axis=1), axis=0)
        
        quality_data_pct.plot(
            kind='bar',
            stacked=True,
            colormap='viridis'
        )
        
        plt.title('Response Quality Distribution by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Percentage')
        plt.legend(title='Response Category', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(output_path / 'quality_distribution.png')
        plt.close()

    def _create_question_performance_plot(self, df: pd.DataFrame, output_path: Path):
        """Create question-wise performance plot."""
        plt.figure(figsize=(15, 8))
        
        question_performance = df.pivot_table(
            index='question',
            columns='model_type',
            values='similarity_score',
            aggfunc='mean'
        )
        
        question_performance.plot(
            kind='bar',
            width=0.8
        )
        
        plt.title('Performance by Question Across Model Types')
        plt.xlabel('Question')
        plt.ylabel('Average Similarity Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(output_path / 'question_performance.png')
        plt.close()

    def _generate_comprehensive_report(self, results: Dict, detailed_df: pd.DataFrame, 
                                    summary_df: pd.DataFrame, output_path: Path):
        """Generate a comprehensive evaluation report."""
        report = "Unified RAG System Evaluation Report\n"
        report += "=================================\n\n"
        report += f"Generated on: {results['timestamp']}\n\n"

        # Overall Performance Summary
        report += self._generate_overall_summary(detailed_df)
        
        # Model Type Comparison
        report += self._generate_model_type_comparison(detailed_df)
        
        # Detailed Model Performance
        report += self._generate_detailed_model_performance(summary_df)
        
        # Question-wise Analysis
        report += self._generate_question_analysis(detailed_df)
        
        # Save the report
        with open(output_path / "comprehensive_evaluation_report.txt", "w", encoding='utf-8') as f:
            f.write(report)

    def _generate_overall_summary(self, df: pd.DataFrame) -> str:
        """Generate overall performance summary section."""
        summary = "\nOverall Performance Summary\n"
        summary += "------------------------\n"
        
        best_model = df.loc[df['similarity_score'].idxmax()]
        fastest_model = df.loc[df['response_time'].idxmin()]
        
        summary += f"Best Performing Model: {best_model['model_id']}\n"
        summary += f"- Similarity Score: {best_model['similarity_score']:.2%}\n"
        summary += f"- Response Time: {best_model['response_time']:.2f} seconds\n\n"
        
        summary += f"Fastest Model: {fastest_model['model_id']}\n"
        summary += f"- Response Time: {fastest_model['response_time']:.2f} seconds\n"
        summary += f"- Similarity Score: {fastest_model['similarity_score']:.2%}\n\n"
        
        return summary

    def _generate_model_type_comparison(self, df: pd.DataFrame) -> str:
        """Generate model type comparison section."""
        comparison = "\nModel Type Comparison\n"
        comparison += "-------------------\n"
        
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            
            comparison += f"\n{model_type.upper()} Models:\n"
            comparison += f"Average Similarity: {model_data['similarity_score'].mean():.2%}\n"
            comparison += f"Average Response Time: {model_data['response_time'].mean():.2f} seconds\n"
            comparison += f"Quality Distribution:\n"
            
            quality_dist = model_data['response_category'].value_counts(normalize=True)
            for category, percentage in quality_dist.items():
                comparison += f"- {category}: {percentage:.1%}\n"
                
        return comparison

    def _generate_detailed_model_performance(self, summary_df: pd.DataFrame) -> str:
        """Generate detailed model performance section."""
        performance = "\nDetailed Model Performance\n"
        performance += "------------------------\n"
        
        for model_id in summary_df.index:
            stats = summary_df.loc[model_id]
            performance += f"\nModel: {model_id}\n"
            
            performance += "Similarity Scores:\n"
            performance += f"- Mean: {stats[('similarity_score', 'mean')]:.2%}\n"
            performance += f"- Std Dev: {stats[('similarity_score', 'std')]:.2%}\n"
            performance += f"- Range: {stats[('similarity_score', 'min')]:.2%} - "
            performance += f"{stats[('similarity_score', 'max')]:.2%}\n"
            
            performance += "Response Times:\n"
            performance += f"- Mean: {stats[('response_time', 'mean')]:.2f}s\n"
            performance += f"- Std Dev: {stats[('response_time', 'std')]:.2f}s\n"
            performance += f"- Range: {stats[('response_time', 'min')]:.2f}s - "
            performance += f"{stats[('response_time', 'max')]:.2f}s\n"
            
        return performance

    def _generate_question_analysis(self, df: pd.DataFrame) -> str:
        """Generate question-wise analysis section."""
        analysis = "\nQuestion-wise Analysis\n"
        analysis += "-------------------\n"
        
        for question in df['question'].unique():
            question_data = df[df['question'] == question]
            best_model = question_data.loc[question_data['similarity_score'].idxmax()]
            
            analysis += f"\nQuestion: {question}\n"
            analysis += f"Best Model: {best_model['model_id']}\n"
            analysis += f"Similarity Score: {best_model['similarity_score']:.2%}\n"
            analysis += f"Response Time: {best_model['response_time']:.2f}s\n"
            
        return analysis

def main():
    """Main execution function."""
    TEST_CASES = [
{
    "question": "What is the process for appealing an academic decision at Northeastern University?",
    "answer": "The academic appeals process at Northeastern University involves several steps:\n1. **Discuss Concerns with Instructor and/or Unit Head**: Initially, students should discuss their concerns with the course instructor. If unresolved, they can meet with the department chair.\n2. **Prepare an Appeal Statement**: Submit a written statement detailing the issue, the decision in question, and the desired resolution within 28 days of the decision.\n3. **Dean's-Level Resolution**: The dean will respond in writing and attempt an informal resolution.\n4. **College-Level Appeal**: If unsatisfied, students can appeal through their college's academic appeals procedure.\n5. **University-Level Appeal**: If still unresolved, students can request a university-level review by the Academic Appeals Resolution Committee.\nFor more details, visit the [Academic Appeals Policies and Procedures](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/academic-appeals-policies-procedures/)."
},
{
    "question": "Who should a student contact if they believe they have been discriminated against during an academic appeal?",
    "answer": "Students who believe they have faced discrimination should contact the Office for University Equity and Compliance (OUEC) as soon as they become aware of the issue. The OUEC will investigate the matter before any academic appeal is pursued. More information can be found on the [OUEC website](https://ouec.northeastern.edu/)."
},
{
    "question": "What are the deadlines for submitting an academic appeal at Northeastern University?",
    "answer": "Students must submit their appeal statement within 28 calendar days from when the academic decision is made available. If disputing a grade in the final term, it must be done within 28 days of the degree conferral date."
},
{
    "question": "Can a student have legal representation during the academic appeal process?",
    "answer": "While students can seek legal advice, a lawyer cannot be present during the informal or formal academic appeal procedures. Students may consult with university officials for guidance."
},
{
    "question": "What happens if a student is not satisfied with the college-level appeal decision?",
    "answer": "If unsatisfied with the college-level decision, students can appeal to the university level by submitting a written request within 10 days of the college's decision. The Academic Appeals Resolution Committee will then review the case."
},
{
    "question": "What is the role of the Academic Appeals Resolution Committee?",
    "answer": "The committee reviews university-level appeals and includes the vice provost for curriculum and programs, a faculty member from the student's college, and two faculty members appointed by the Faculty Senate Agenda Committee. They conduct investigations and make final decisions on appeals."
},
{
    "question": "What should be included in an appeal statement?",
    "answer": "An appeal statement should include:\n- The date of the issue\n- The decision being disputed\n- The nature of the decision\n- The desired resolution\n- Any supporting materials\nThis statement must be submitted within 28 days of the decision."
},
{
    "question": "What is the role of the Office for University Equity and Compliance in the appeals process?",
    "answer": "The OUEC investigates claims of harassment or discrimination related to academic appeals. They must be consulted before the academic appeal proceeds if such issues are involved. Visit the [OUEC website](https://ouec.northeastern.edu/) for more information."
},
{
    "question": "What actions can the Academic Appeals Resolution Committee take?",
    "answer": "The committee can uphold the college's decision or grant relief to the student. They cannot contradict prior findings of the OUEC. Their decision is final, and no further appeal is possible."
},
{
    "question": "How are parties informed of the appeal decision?",
    "answer": "All parties involved, including the student, faculty member, and relevant university officials, will be informed in writing of the decisions and actions taken during the appeals process."
},
{
    "question": "What are the general attendance requirements at Northeastern University?",
    "answer": "Class participation is essential for success in any course format. Each instructor may have specific attendance policies, and it is the student's responsibility to understand these requirements. Failure to meet attendance requirements may result in the need to drop courses. Students should avoid conflicting commitments until class schedules are finalized. Permission to make up work may be granted for reasonable causes, and requests should be made immediately upon returning to class. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "How should a student handle absences due to university-sponsored activities?",
    "answer": "Students participating in university-sponsored activities, such as athletic competitions or research presentations, may have excused absences. These absences should be discussed with instructors at least two weeks in advance. Instructors may require a written statement from the activity's administrator. Students should develop a plan to make up missed coursework with their instructors. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "What is the policy for absences due to religious beliefs?",
    "answer": "Students unable to attend classes or participate in exams due to religious beliefs should be allowed to make up missed work. Arrangements should be made with the instructor at least two weeks before the religious observance. The makeup work should not create an unreasonable burden on the university. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "How should a student manage absences due to jury duty?",
    "answer": "Students called for jury duty should inform their instructors, who will provide reasonable substitute or compensatory opportunities for missed work. Students will not be penalized for such absences. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "What should a student do if they have an unforeseen absence?",
    "answer": "In the event of unforeseen circumstances, such as illness, students should notify their instructors and academic advisor as soon as possible. They should work with instructors to develop a plan to make up missed coursework. Medical documentation is not required. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "What are the procedures for extended absences?",
    "answer": "Students absent for an extended period should inform their academic advisor. Depending on the absence's length, students may need to apply for a medical or emergency leave of absence. It is recommended to discuss potential next steps with an academic advisor. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "What are the consequences of nonattendance?",
    "answer": "Nonattendance does not equate to officially dropping or withdrawing from a course. Students are responsible for the academic and financial consequences. Grades earned due to nonattendance affect academic progression, visa eligibility for international students, and federal financial aid eligibility. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "What should a student do if they need to make up laboratory work?",
    "answer": "Laboratory work can only be made up during regularly scheduled instruction hours. Students should seek permission from instructors to make up work for reasonable causes. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "How can a student request permission to make up missed work?",
    "answer": "Students should request permission to make up missed work immediately upon returning to class. Instructors may grant permission for reasonable causes. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "What are the expectations for instructors regarding student absences?",
    "answer": "Instructors are expected to make reasonable accommodations for excused absences, including makeup assignments and exams. They may require documentation for certain absences, such as university-sponsored activities. [More details](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/attendance-requirements/)."
},
{
    "question": "What are the general graduation requirements at Northeastern University?",
    "answer": "To be eligible for a degree, students must meet all academic, program-specific, residency, and good standing requirements. They must also clear any financial, experiential education, and disciplinary deficiencies. More details can be found [here](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/graduation-requirements/)."
},
{
    "question": "Where can I find the specific program of study requirements for my degree?",
    "answer": "Specific program requirements are detailed under each program in the Northeastern University Academic Catalog. Visit the catalog [here](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/graduation-requirements/) for more information."
},
{
    "question": "Can I take courses at another institution to fulfill my degree requirements?",
    "answer": "Students may petition their college to take courses at another accredited institution to clear deficiencies or access unavailable courses. Approval is required from respective advisors and colleges."
},
{
    "question": "What is the residency requirement for a bachelor's degree at Northeastern University?",
    "answer": "Students must earn a minimum of 64 Northeastern semester hours. Exchange programs and specialized programs may have different requirements. More details are available in the Academic Catalog."
},
{
    "question": "What are the Universitywide Requirements for undergraduate students?",
    "answer": "All undergraduate students must complete the Universitywide Requirements as part of their degree program. These are detailed in the Northeastern University Academic Catalog."
},
{
    "question": "What are the NUpath requirements?",
    "answer": "NUpath requirements are mandatory for all undergraduate students. They are designed to ensure a comprehensive educational experience. More information can be found in the Academic Catalog."
},
{
    "question": "Is attendance at Commencement mandatory?",
    "answer": "Attendance at Commencement is optional. Graduating seniors receive information during the spring semester. Students who do not qualify for their degrees are notified."
},
{
    "question": "How can I ensure I am on track for graduation?",
    "answer": "Students should complete a graduation degree audit at their college's academic advising office before finishing their program."
},
{
    "question": "What should I do if I have been removed from the graduation list?",
    "answer": "Students removed from the graduation list are notified if they fail to qualify for their degrees. It is advisable to contact your academic advisor for further guidance."
},
{
    "question": "Where can I find more information about graduation policies and procedures?",
    "answer": "Detailed information about graduation policies and procedures is available in the Northeastern University Academic Catalog. Visit the catalog [here](https://catalog.northeastern.edu/undergraduate/academic-policies-procedures/graduation-requirements/)."
},
{
    "question": "What are the immunization requirements for students enrolling at Northeastern University?",
    "answer": "Massachusetts state law mandates that all students provide proof of immunity to certain diseases before arriving at the university. For students enrolling in fall 2024, Northeastern University partners with Sentry MD to manage health record compliance. All incoming students must submit the required vaccination documentation to Sentry MD by July 31, 2024. Failure to comply will result in prohibition from registering and attending classes. For more information, visit the [University Health and Counseling Services website](https://catalog.northeastern.edu/undergraduate/information-entering-students/health-requirements-uhcs/) or contact Sentry MD at [Northeastern.immunizations@sentrymd.com](mailto:Northeastern.immunizations@sentrymd.com)."
},
{
    "question": "How can I submit my immunization documentation to Northeastern University?",
    "answer": "Students must submit their immunization documentation to Sentry MD, a confidential health record compliance service, by July 31, 2024. To learn more and submit your information, review \"Sentry MD\" in the [Student Hub](https://me.northeastern.edu) under Health and Wellness. For assistance, contact Sentry MD at [Northeastern.immunizations@sentrymd.com](mailto:Northeastern.immunizations@sentrymd.com)."
},
{
    "question": "What happens if I fail to provide the required immunization documentation?",
    "answer": "Students who do not submit the required immunization documentation by the deadline will be prohibited from registering and attending all classes. Ensure compliance by submitting your documents to Sentry MD by July 31, 2024. For questions, contact Sentry MD at [Northeastern.immunizations@sentrymd.com](mailto:Northeastern.immunizations@sentrymd.com)."
},
{
    "question": "Are there any immunization requirements for students at nonresidential campuses?",
    "answer": "Students enrolled at a nonresidential campus are not required to submit proof of immunity. However, it is recommended that they keep a copy of their immunization records in their files for personal reference."
},
{
    "question": "Where can I find more information about health requirements for entering students at Northeastern University?",
    "answer": "Detailed information about health requirements for entering students can be found on the [University Health and Counseling Services website](https://catalog.northeastern.edu/undergraduate/information-entering-students/health-requirements-uhcs/). For further inquiries, contact Sentry MD at [Northeastern.immunizations@sentrymd.com](mailto:Northeastern.immunizations@sentrymd.com)."
},
{
    "question": "Who should I contact if I have questions about immunization requirements at Northeastern University?",
    "answer": "For questions regarding immunization requirements or submitting documentation, contact Sentry MD at [Northeastern.immunizations@sentrymd.com](mailto:Northeastern.immunizations@sentrymd.com)."
},
{
    "question": "What is the deadline for submitting immunization documentation for fall 2024 enrollment?",
    "answer": "The deadline for submitting immunization documentation for students enrolling in fall 2024 is July 31, 2024. Ensure your documents are submitted to Sentry MD by this date to avoid registration and attendance issues."
},
{
    "question": "What is Sentry MD, and how does it relate to Northeastern University's immunization process?",
    "answer": "Sentry MD is a confidential health record compliance service partnered with Northeastern University to streamline the document review and tracking process for immunization requirements. This service helps students manage their health requirements efficiently. For more information, visit the [Student Hub](https://me.northeastern.edu) under Health and Wellness."
},
{
    "question": "Is there a specific email address for immunization-related inquiries at Northeastern University?",
    "answer": "Yes, for immunization-related inquiries, you can contact Sentry MD at [Northeastern.immunizations@sentrymd.com](mailto:Northeastern.immunizations@sentrymd.com)."
},
{
    "question": "What is the Student Hub, and how is it related to health and wellness at Northeastern University?",
    "answer": "The Student Hub is an online platform where students can access various resources, including health and wellness information. To learn more about submitting immunization information, review \"Sentry MD\" in the [Student Hub](https://me.northeastern.edu) under Health and Wellness."
},
{
    "question": "What is the Northeastern University Police Department (NUPD) and what services do they offer?",
    "answer": "The Northeastern University Police Department (NUPD) is a full-service, accredited police agency providing 24-hour patrol and investigative services. They focus on crime detection and prevention through technology and community engagement. Services include personal safety escorts, the RedEye nighttime off-campus escort service, and the SafeZone mobile safety app. For more information, visit the [NUPD website](https://nupd.northeastern.edu/)."
},
{
    "question": "How can I contact the Northeastern University Police Department in case of an emergency?",
    "answer": "In case of an emergency, you can contact the Northeastern University Police Department at 617.373.3333. For emergencies involving the deaf and hearing impaired, dial 711. For non-emergencies, call 617.373.2121. More details are available on the [NUPD website](https://nupd.northeastern.edu/)."
},
{
    "question": "What is the RedEye service and how can I use it?",
    "answer": "The RedEye service is a nighttime off-campus escort service that operates from dusk to dawn, transporting students to their residences within two miles of the Boston campus. To use this service, book a ride in advance using the RedEye app or at the RedEye dispatch center located at the Northeast Security office in the Ruggles Substation. The service runs every night from 5 p.m. to 6 a.m."
},
{
    "question": "What is the SafeZone app and how does it work?",
    "answer": "SafeZone is a mobile safety app available to Northeastern students and staff. It connects users directly to the NUPD for assistance or emergency support while on campus. The app is free to download and use. For more information, visit the [SafeZone page](https://nupd.northeastern.edu/safezone/)."
},
{
    "question": "How can I report a lost item on the Boston campus?",
    "answer": "To report a lost item on the Boston campus, call 617.373.3913. If your item is found, you will be contacted by telephone or email. If you find an item of value, return it to the NUPD headquarters at 100 Columbus Place. If you suspect theft, report it to the NUPD at 617.373.2121."
},
{
    "question": "What should I do in case of a snow emergency at Northeastern University?",
    "answer": "In case of a snow emergency, contact Northeastern University's emergency line at 617.373.2000 for updates and instructions. For other emergencies, call 617.373.3333."
},
{
    "question": "What is the procedure for fire egress drills in residence halls?",
    "answer": "Fire egress drills are conducted each semester in residence halls to familiarize students with the alarm system and evacuation routes. Participation is mandatory for all building occupants. For fire safety tips, visit the [NUPD website](https://nupd.northeastern.edu/safety/emergency-planning/)."
},
{
    "question": "How does Northeastern University handle emergency situations?",
    "answer": "Northeastern University has a trained team of police officers, EMTs, health and counseling experts, and other professionals to manage emergencies. The NU Alert system sends emergency broadcast messages to students, faculty, and staff. For more information, visit the [NUPD website](https://nupd.northeastern.edu/safety/emergency-planning/)."
},
{
    "question": "What are the contact details for the Oakland Department of Public Safety?",
    "answer": "The Oakland Department of Public Safety can be reached at 510.430.3333 for campus emergencies and 510.430.5555 for safety and transportation inquiries. For more information, email Oaklandsafety@northeastern.edu or visit their [website](https://oakland.northeastern.edu/student-resources/campus-safety/)."
},
{
    "question": "How can I access personal safety services at Northeastern University?",
    "answer": "For personal safety services, contact the NUPD at 617.373.2121. They offer safety escorts and the RedEye service. More details are available on the [Personal Safety page](https://nupd.northeastern.edu/our-services/safety-escort-services/)."
}

]
    try:
        # Setup
        Config.setup()
        
        # Create model configurations
        model_configs = [
            # LLaMA configurations
            ModelConfig(
                model_type="llama",
                model_name="meta-llama/Llama-2-7b-chat-hf",
                embedding_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
            ModelConfig(
                model_type="llama",
                model_name="meta-llama/Llama-2-13b-chat-hf",
                embedding_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
            ModelConfig(
                model_type="llama",
                model_name="meta-llama/Llama-3.2-3B",
                embedding_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
            # GPT configurations
            ModelConfig(
                model_type="gpt",
                model_name="gpt-4",
                embedding_name="text-embedding-3-large"
            ),
            ModelConfig(
                model_type="gpt",
                model_name="gpt-3.5-turbo",
                embedding_name="text-embedding-3-small"
            )
        ]
        
        # Initialize evaluator
        logger.info(f"Starting unified evaluation with {len(model_configs)} model configurations")
        evaluator = UnifiedModelEvaluator(TEST_CASES, model_configs)
        
        # Run evaluation
        results = evaluator.evaluate_all_models("../../refined")
        
        # Save results
        logger.info("Saving evaluation results...")
        evaluator.save_results(results, Config.OUTPUT_DIR)
        
        logger.info("Unified evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error during unified evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()