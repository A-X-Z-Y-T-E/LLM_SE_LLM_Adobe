import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import os
import sys
from sklearn.neighbors import NearestNeighbors
import torch_geometric
from torch_geometric.data import Data
from datetime import datetime

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from graph_utils import save_graph_to_disk, load_combined_graphs
except ImportError:
    print("⚠️ graph_utils not found - some features may not work")
    save_graph_to_disk = None
    load_combined_graphs = None

# ========================================================================
# GRAPH CONSTRUCTION FOR DOCUMENT LAYOUT ANALYSIS
# ========================================================================

def build_document_graph(all_text_blocks_processed: list, auto_save: bool = True, 
                        output_dir: str = None) -> Tuple[Optional[Data], List[Dict]]:
    """
    Build a PyTorch Geometric graph from processed text blocks for Document Layout Analysis.
    
    Args:
        all_text_blocks_processed (list): List of processed text block dictionaries
        auto_save (bool): Whether to automatically save the graph to disk
        output_dir (str): Directory to save graphs (default: training_data/graphs)
        
    Returns:
        Tuple containing:
            - pyg_data_object: torch_geometric.data.Data object or None if empty input
            - original_blocks: Original input list for mapping predictions back
    """
    # Handle empty input
    if not all_text_blocks_processed:
        print("⚠️ Empty input provided to build_document_graph")
        return None, []
    
    num_nodes = len(all_text_blocks_processed)
    print(f"🔧 Building graph with {num_nodes} nodes...")
    
    # Step 1: Extract features and labels
    print("📊 Extracting node features and labels...")
    node_features, node_labels = extract_node_features_and_labels(all_text_blocks_processed)
    
    # Step 2: Build edges
    print("🔗 Building graph edges...")
    edge_index = build_graph_edges(all_text_blocks_processed)
    
    # Step 3: Create PyTorch Geometric Data object
    pyg_data = Data(
        x=node_features,           # [num_nodes, 22] features
        y=node_labels,             # [num_nodes] labels
        edge_index=edge_index,     # [2, num_edges] edge connections
        num_nodes=num_nodes
    )
    
    print(f"✅ Graph constructed successfully:")
    print(f"   📊 Nodes: {pyg_data.num_nodes}")
    print(f"   🔗 Edges: {pyg_data.num_edges}")
    print(f"   📐 Features: {pyg_data.x.shape}")
    print(f"   🏷️  Labels: {pyg_data.y.shape}")
    
    # Auto-save if requested
    if auto_save and pyg_data is not None and save_graph_to_disk is not None:
        try:
            if output_dir is None:
                project_root = Path(__file__).parent.parent
                output_dir = project_root / "training_data" / "graphs"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename from first block's pdf_id or timestamp
            pdf_id = all_text_blocks_processed[0].get('pdf_id', 'document')
            safe_pdf_id = "".join(c for c in pdf_id if c.isalnum() or c in ('_', '-'))
            graph_filename = f"{safe_pdf_id}_graph.pt"
            graph_path = output_dir / graph_filename
            
            save_graph_to_disk(pyg_data, all_text_blocks_processed, str(graph_path))
            
        except Exception as e:
            print(f"⚠️ Auto-save failed: {e}")
    
    return pyg_data, all_text_blocks_processed

def extract_node_features_and_labels(blocks: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract node features and labels from text blocks.
    
    Args:
        blocks (List[Dict]): List of text block dictionaries
        
    Returns:
        Tuple of (features_tensor, labels_tensor)
    """
    features_list = []
    labels_list = []
    
    for block in blocks:
        # Extract features - check different possible field names
        if 'feature_vectors' in block:
            features = block['feature_vectors']
        elif 'features' in block:
            # If features is a dict, convert to list in consistent order
            features_dict = block['features']
            if isinstance(features_dict, dict):
                features = extract_features_from_dict(features_dict)
            else:
                features = features_dict
        else:
            # Create default features if missing
            features = [0.0] * 22  # 22 features as specified
            print(f"⚠️ Missing features for block {block.get('block_id', 'unknown')}")
        
        # Ensure features is a list of correct length
        if len(features) != 22:
            print(f"⚠️ Expected 22 features, got {len(features)} for block {block.get('block_id', 'unknown')}")
            # Pad or truncate to 22
            features = (features + [0.0] * 22)[:22]
        
        # Extract label
        label_id = block.get('label_id', 0)  # Default to BODY (0) if missing
        
        features_list.append(features)
        labels_list.append(label_id)
    
    # Convert to tensors
    features_tensor = torch.tensor(features_list, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    
    return features_tensor, labels_tensor

def extract_features_from_dict(features_dict: Dict[str, float]) -> List[float]:
    """
    Extract features from dictionary in consistent order.
    
    Args:
        features_dict (Dict[str, float]): Features as dictionary
        
    Returns:
        List[float]: Features in consistent order (22 total)
    """
    # Define the expected feature order (22 total features)
    feature_order = [
        # Geometric features (8)
        "x0_norm", "y0_norm", "x1_norm", "y1_norm", "width_norm", "height_norm", 
        "is_left_aligned", "is_centered_horizontally",
        
        # Stylistic features (3)
        "font_size_norm", "is_bold", "is_italic",
        
        # Textual features (5)
        "text_length_chars_norm", "starts_with_bullet", "ends_with_colon", 
        "is_all_caps", "contains_number_prefix",
        
        # Contextual features (6)
        "y_offset_to_prev_block_norm", "x_offset_to_prev_block_norm", 
        "font_size_ratio_to_prev_block_norm", "is_prev_block_same_font_size", 
        "is_prev_block_same_bold_status", "is_prev_block_same_indentation"
    ]
    
    features = []
    for feature_name in feature_order:
        features.append(float(features_dict.get(feature_name, 0.0)))
    
    # Verify we have exactly 22 features
    assert len(features) == 22, f"Expected 22 features but got {len(features)}"
    
    return features

def build_graph_edges(blocks: List[Dict]) -> torch.Tensor:
    """
    Build graph edges including reading order and spatial KNN edges.
    
    Args:
        blocks (List[Dict]): List of text block dictionaries
        
    Returns:
        torch.Tensor: Edge index tensor of shape [2, num_edges]
    """
    num_nodes = len(blocks)
    edges = []
    
    # 1. Reading Order Edges (Sequential)
    print("   📖 Adding reading order edges...")
    reading_edges = build_reading_order_edges(num_nodes)
    edges.extend(reading_edges)
    
    # 2. Within-Page K-Nearest Neighbors (KNN) Edges
    print("   🎯 Adding KNN spatial edges...")
    knn_edges = build_knn_spatial_edges(blocks, k=3)
    edges.extend(knn_edges)
    
    # Convert to tensor
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Create empty edge index if no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    print(f"   🔗 Total edges: {edge_index.shape[1]}")
    
    return edge_index

def build_reading_order_edges(num_nodes: int) -> List[Tuple[int, int]]:
    """
    Build sequential reading order edges.
    
    Args:
        num_nodes (int): Number of nodes
        
    Returns:
        List[Tuple[int, int]]: List of (source, target) edge pairs
    """
    edges = []
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
    
    print(f"      📖 Created {len(edges)} reading order edges")
    return edges

def build_knn_spatial_edges(blocks: List[Dict], k: int = 3) -> List[Tuple[int, int]]:
    """
    Build K-nearest neighbor edges based on spatial proximity within pages.
    
    Args:
        blocks (List[Dict]): List of text block dictionaries
        k (int): Number of nearest neighbors
        
    Returns:
        List[Tuple[int, int]]: List of (source, target) edge pairs
    """
    edges = []
    
    # Group blocks by page
    pages = {}
    for idx, block in enumerate(blocks):
        page_num = block.get('page_number', 1)
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append((idx, block))
    
    # Process each page separately
    for page_num, page_blocks in pages.items():
        if len(page_blocks) <= 1:
            continue  # Skip pages with only one block
        
        # Extract center points
        indices = [idx for idx, _ in page_blocks]
        centers = []
        
        for _, block in page_blocks:
            bbox = block.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
            else:
                center_x, center_y = 0, 0
            centers.append([center_x, center_y])
        
        # Skip if not enough blocks for KNN
        if len(centers) <= k:
            # Connect all blocks to all other blocks on this page
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    if i != j:  # Avoid self-loops
                        edges.append((idx_i, idx_j))
        else:
            # Use KNN to find neighbors
            centers_np = np.array(centers)
            nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(centers)), algorithm='kd_tree')
            nbrs.fit(centers_np)
            
            # Find neighbors for each block
            for i, center in enumerate(centers):
                distances, neighbor_indices = nbrs.kneighbors([center])
                
                # Add edges to k nearest neighbors (excluding self)
                for neighbor_idx in neighbor_indices[0]:
                    if neighbor_idx != i:  # Avoid self-loops
                        source_global_idx = indices[i]
                        target_global_idx = indices[neighbor_idx]
                        edges.append((source_global_idx, target_global_idx))
    
    print(f"      🎯 Created {len(edges)} KNN spatial edges")
    return edges

def load_processed_document_to_graph(json_file_path: str, save_graph: bool = False, 
                                   output_dir: str = None) -> Tuple[Optional[Data], List[Dict]]:
    """
    Load a processed document JSON file and convert it to a graph.
    
    Args:
        json_file_path (str): Path to processed document JSON file
        save_graph (bool): Whether to save the built graph to disk
        output_dir (str): Directory to save graphs (default: training_data/graphs)
        
    Returns:
        Tuple of (graph_data, original_blocks)
    """
    file_path = Path(json_file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found: {json_file_path}")
    
    print(f"📄 Loading document: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract blocks from different possible structures
        if 'blocks' in data:
            blocks = data['blocks']
        elif isinstance(data, list):
            blocks = data
        else:
            raise ValueError("Unexpected JSON structure - no 'blocks' field found")
        
        if not blocks:
            print("⚠️ No blocks found in document")
            return None, []
        
        # Build graph
        graph_data, original_blocks = build_document_graph(blocks, auto_save=False)
        
        # Save graph if requested
        if save_graph and graph_data is not None and save_graph_to_disk is not None:
            if output_dir is None:
                project_root = Path(__file__).parent.parent
                output_dir = project_root / "training_data" / "graphs"
            
            output_dir = Path(output_dir)
            graph_filename = f"{file_path.stem}_graph.pt"
            graph_path = output_dir / graph_filename
            
            save_graph_to_disk(graph_data, original_blocks, str(graph_path))
    
        return graph_data, original_blocks
        
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        raise

def batch_process_documents_to_graphs(documents_directory: str, save_graphs: bool = True,
                                    output_dir: str = None, 
                                    save_combined: bool = True) -> List[Tuple[Data, List[Dict], str]]:
    """
    Process multiple documents to graphs with automatic saving options.
    
    Args:
        documents_directory (str): Directory containing processed document JSON files
        save_graphs (bool): Whether to save each individual graph to disk
        output_dir (str): Directory to save graphs
        save_combined (bool): Whether to save all graphs in a combined file
        
    Returns:
        List of (graph_data, original_blocks, file_path) tuples
    """
    directory = Path(documents_directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {documents_directory}")
    
    # Find processed document files
    json_files = list(directory.glob("*_processed_features.json"))
    
    if not json_files:
        print(f"⚠️ No *_processed_features.json files found in: {documents_directory}")
        return []
    
    print(f"🔧 Processing {len(json_files)} documents to graphs...")
    print(f"   💾 Save individual graphs: {save_graphs}")
    print(f"   📦 Save combined graphs: {save_combined}")
    
    graphs = []
    successful = 0
    failed = 0
    
    if save_graphs or save_combined:
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "training_data" / "graphs"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for json_file in json_files:
        try:
            # Load without auto-save to control saving here
            graph_data, original_blocks = load_processed_document_to_graph(str(json_file), save_graph=False)
            
            if graph_data is not None:
                # Save individual graph if requested
                if save_graphs and save_graph_to_disk is not None:
                    graph_filename = f"{json_file.stem}_graph.pt"
                    graph_path = output_dir / graph_filename
                    save_graph_to_disk(graph_data, original_blocks, str(graph_path))
                
                graphs.append((graph_data, original_blocks, str(json_file)))
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"❌ Failed to process {json_file.name}: {e}")
            failed += 1
    
    # Save combined graphs if requested
    if save_combined and graphs:
        try:
            combined_path = output_dir / "combined_graphs.pt"
            
            # Prepare combined data
            combined_data = {
                'graphs': graphs,
                'metadata': {
                    'total_documents': len(graphs),
                    'total_nodes': sum(g[0].num_nodes for g in graphs),
                    'total_edges': sum(g[0].num_edges for g in graphs),
                    'processing_timestamp': datetime.now().isoformat()
                }
            }
            
            # Save with explicit protocol for compatibility
            torch.save(combined_data, combined_path)
            print(f"📦 Saved combined graphs to: {combined_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save combined graphs: {e}")
    
    print(f"✅ Graph processing complete:")
    print(f"   📊 Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    
    return graphs

# ========================================================================
# EXAMPLE USAGE FUNCTIONS
# ========================================================================

def example_single_document_graph():
    """Example of building a graph from a single document."""
    print("🔍 EXAMPLE: Building graph from single document")
    print("=" * 50)
    
    # Path to a processed document file
    doc_file = "training_data/processed_by_document/0_processed_features.json"
    
    try:
        # Load and build graph
        graph_data, original_blocks = load_processed_document_to_graph(doc_file)
        
        if graph_data is not None:
            print(f"\n📈 Graph Statistics:")
            print(f"   Nodes: {graph_data.num_nodes}")
            print(f"   Edges: {graph_data.num_edges}")
            print(f"   Features shape: {graph_data.x.shape}")
            print(f"   Labels shape: {graph_data.y.shape}")
            print(f"   Edge density: {graph_data.num_edges / (graph_data.num_nodes * (graph_data.num_nodes - 1)):.4f}")
        
    except Exception as e:
        print(f"❌ Error in example: {e}")

def example_batch_processing(save_graphs: bool = True, save_combined: bool = True, output_dir: str = None):
    """Example of batch processing documents to graphs."""
    print("🔍 EXAMPLE: Batch processing documents to graphs")
    print("=" * 50)
    
    # Directory containing processed documents
    docs_dir = "training_data/processed_by_document"
    
    try:
        # Process all documents
        graphs = batch_process_documents_to_graphs(
            docs_dir, 
            save_graphs=save_graphs, 
            save_combined=save_combined,
            output_dir=output_dir
        )
        
        if graphs:
            print(f"\n📈 Batch Statistics:")
            total_nodes = sum(graph.num_nodes for graph, _, _ in graphs)
            total_edges = sum(graph.num_edges for graph, _, _ in graphs)
            print(f"   Documents: {len(graphs)}")
            print(f"   Total nodes: {total_nodes}")
            print(f"   Total edges: {total_edges}")
            print(f"   Avg nodes/doc: {total_nodes/len(graphs):.1f}")
            print(f"   Avg edges/doc: {total_edges/len(graphs):.1f}")
        
    except Exception as e:
        print(f"❌ Error in example: {e}")

def example_multiple_documents():
    """Example of loading multiple documents and converting to tensors."""
    print("🔍 EXAMPLE: Loading multiple documents to tensors")
    print("=" * 50)
    
    try:
        # Check if combined graphs exist
        combined_path = "training_data/graphs/combined_graphs.pt"
        if Path(combined_path).exists() and load_combined_graphs is not None:
            graphs, metadata = load_combined_graphs(combined_path)
            
            print(f"📊 Loaded {len(graphs)} graphs from combined file")
            print(f"📈 Total nodes: {metadata.get('total_nodes', 'unknown')}")
            print(f"🔗 Total edges: {metadata.get('total_edges', 'unknown')}")
            
            # Show sample graph info
            if graphs:
                sample_graph, sample_blocks, sample_path = graphs[0]
                print(f"\n📄 Sample graph from: {Path(sample_path).name}")
                print(f"   Nodes: {sample_graph.num_nodes}")
                print(f"   Edges: {sample_graph.num_edges}")
                print(f"   Features: {sample_graph.x.shape}")
                print(f"   Labels: {sample_graph.y.shape}")
        else:
            print("❌ Combined graphs file not found. Run batch processing first.")
            print("   Use option 6 to auto-process all documents.")
            
    except Exception as e:
        print(f"❌ Error in example: {e}")

def build_graphs_from_features_directory(features_directory: str = None, 
                                       output_directory: str = None) -> List[Tuple[Data, List[Dict], str]]:
    """
    Build graphs from all processed features in a directory.
    
    Args:
        features_directory (str): Directory containing processed features JSON files
        output_directory (str): Directory to save graphs
        
    Returns:
        List of (graph_data, original_blocks, file_path) tuples
    """
    if features_directory is None:
        project_root = Path(__file__).parent.parent
        features_directory = project_root / "training_data" / "processed_by_document"
    
    if output_directory is None:
        project_root = Path(__file__).parent.parent
        output_directory = project_root / "training_data" / "graphs"
    
    print(f"🔧 Building graphs from features directory: {features_directory}")
    
    return batch_process_documents_to_graphs(
        documents_directory=str(features_directory),
        save_graphs=True,
        output_dir=str(output_directory),
        save_combined=True
    )

def quick_graph_build():
    """Quick function to build all graphs with default settings."""
    print("🚀 QUICK GRAPH BUILD - Building all graphs with default settings")
    print("=" * 60)
    
    try:
        # Build from default directories
        graphs = build_graphs_from_features_directory()
        
        if graphs:
            print(f"\n✅ Successfully built {len(graphs)} graphs!")
            print(f"   📁 Saved to: training_data/graphs/")
            
            # Show summary statistics
            total_nodes = sum(g[0].num_nodes for g in graphs)
            total_edges = sum(g[0].num_edges for g in graphs)
            
            print(f"\n📊 Summary Statistics:")
            print(f"   Total documents: {len(graphs)}")
            print(f"   Total nodes: {total_nodes}")
            print(f"   Total edges: {total_edges}")
            print(f"   Average nodes per document: {total_nodes/len(graphs):.1f}")
            print(f"   Average edges per document: {total_edges/len(graphs):.1f}")
            
            # Verify files were created
            graphs_dir = Path("training_data/graphs")
            if graphs_dir.exists():
                graph_files = list(graphs_dir.glob("*_graph.pt"))
                combined_file = graphs_dir / "combined_graphs.pt"
                
                print(f"\n📁 Files created:")
                print(f"   Individual graphs: {len(graph_files)}")
                print(f"   Combined file: {'✅' if combined_file.exists() else '❌'}")
            
            return graphs
        else:
            print("❌ No graphs were built. Check your processed features directory.")
            return []
            
    except Exception as e:
        print(f"❌ Quick graph build failed: {e}")
        import traceback
        traceback.print_exc()
        return []

# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    print("🚀 GRAPH CONSTRUCTION FOR DOCUMENT LAYOUT ANALYSIS")
    print("=" * 70)
    
    # Choose what to run
    print("Choose an option:")
    print("1. Quick build - Auto-process all documents (RECOMMENDED)")
    print("2. Load multiple document vectors (tensor example)")
    print("3. Build graph from single document")
    print("4. Batch process documents to graphs (custom options)")
    print("5. Convert specific processed document file")
    print("6. Auto-process all documents with default settings")
    print("7. View existing graphs statistics")
    
    choice = input("Enter choice (1-7): ").strip()
    
    if choice == "1":
        quick_graph_build()
        
    elif choice == "2":
        example_multiple_documents()
        
    elif choice == "3":
        example_single_document_graph()
        
    elif choice == "4":
        save_individual = input("Save individual graphs? (y/n, default: y): ").strip().lower() != 'n'
        save_combined = input("Save combined graphs file? (y/n, default: y): ").strip().lower() != 'n'
        
        if save_individual or save_combined:
            output_dir = input("Output directory (default: training_data/graphs): ").strip()
            if not output_dir:
                output_dir = None
        else:
            output_dir = None
            
        example_batch_processing(save_graphs=save_individual, save_combined=save_combined, output_dir=output_dir)
        
    elif choice == "5":
        file_path = input("Enter path to processed document JSON file: ").strip()
        save_graph = input("Save graph to disk? (y/n, default: y): ").strip().lower() != 'n'
        
        try:
            graph_data, original_blocks = load_processed_document_to_graph(file_path, save_graph=save_graph)
            print("✅ Graph construction successful!")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    elif choice == "6":
        print("🔄 Auto-processing with default settings...")
        print("   💾 Saving individual graphs: ✅")
        print("   📦 Saving combined file: ✅")
        print("   📁 Output: training_data/graphs/")
        
        try:
            graphs = build_graphs_from_features_directory()
            
            if graphs:
                print(f"\n🎉 Processing complete!")
                print(f"   📊 Documents processed: {len(graphs)}")
                print(f"   💾 Graphs saved to: training_data/graphs/")
                
        except Exception as e:
            print(f"❌ Auto-processing failed: {e}")
    
    elif choice == "7":
        # View existing graphs
        try:
            combined_path = "training_data/graphs/combined_graphs.pt"
            if Path(combined_path).exists() and load_combined_graphs is not None:
                graphs, metadata = load_combined_graphs(combined_path)
                print(f"📊 Found {len(graphs)} existing graphs")
            else:
                print("❌ No existing graphs found. Run option 1 or 6 first.")
        except Exception as e:
            print(f"❌ Error viewing graphs: {e}")
    
    else:
        print("Invalid choice. Running quick build...")
        quick_graph_build()
    
    print("\n🎉 Graph operations complete!")