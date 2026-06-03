import argparse
import yaml
import json
import os
from pathlib import Path
from aligner.database import Database
from aligner.file_loader import FileLoader
from aligner.embedding_engine import EmbeddingEngine
from aligner.chapter_fingerprint import ChapterFingerprint
from aligner.similarity_matrix import SimilarityMatrix
from aligner.sequence_aligner import SequenceAligner
from aligner.graph_builder import GraphBuilder

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_alignment(base_path: str, config: dict):
    novel_name = os.path.basename(os.path.normpath(base_path))
    db = Database(config['database']['path'])
    novel_id = db.get_or_create_novel(novel_name)

    loader = FileLoader()
    jp_files = loader.load_chapters(base_path, "JP")
    en_files = loader.load_chapters(base_path, "EN")

    engine = EmbeddingEngine(config['embedding']['model'], config['embedding']['device'])
    fingerprinter = ChapterFingerprint(engine, config['embedding']['max_characters'], config['embedding']['mode'])

    # Fingerprinting & Caching
    def get_fps(files, lang):
        fps = []
        for file in files:
            fp = db.get_fingerprint(novel_id, lang, file)
            if not fp:
                abs_path = os.path.join(base_path, file)
                fp = fingerprinter.extract(file, abs_path)
                db.save_fingerprint(novel_id, lang, file, fp)
            fps.append(fp)
        return fps

    print("Generating/Loading fingerprints...")
    jp_fps = get_fps(jp_files, "JP")
    en_fps = get_fps(en_files, "EN")

    print("Computing similarity matrix...")
    matrix_builder = SimilarityMatrix(config['similarity_weights'])
    sim_matrix = matrix_builder.build_matrix(jp_fps, en_fps)

    print("Aligning sequences via Dynamic Programming...")
    aligner = SequenceAligner(config['alignment']['skip_penalty'], config['alignment']['match_threshold'])
    alignments = aligner.align(sim_matrix)

    print("Building JSON Graph...")
    graph = GraphBuilder.build_graph(novel_name, jp_files, en_files, alignments)

    # Export Graph
    out_path = f"{novel_name}_alignment_graph.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Validation Report:\n{json.dumps(graph['stats'], indent=2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["align", "rebuild-embeddings", "export-graph"])
    parser.add_argument("path", help="Path to the novel directory")
    args = parser.parse_args()

    config = load_config()
    
    if args.command == "align":
        run_alignment(args.path, config)