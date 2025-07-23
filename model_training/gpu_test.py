"""
GPU and CUDA testing script for RTX 4050.
Tests PyTorch, PyTorch Geometric, and GNN operations.
"""

import torch
import sys
import time
import traceback
from pathlib import Path


def test_pytorch_basics():
    """Test basic PyTorch functionality."""
    print("🔍 TESTING PYTORCH 2.5.0 BASICS")
    print("=" * 30)
    
    try:
        # Basic info
        print(f"🚀 PyTorch version: {torch.__version__}")
        print(f"🐍 Python version: {sys.version}")
        print(f"⚡ CUDA available: {torch.cuda.is_available()}")
        
        # Check if we have PyTorch 2.5.0
        if torch.__version__.startswith('2.5'):
            print("✅ PyTorch 2.5.0 detected - Latest optimizations enabled!")
        else:
            print(f"⚠️ PyTorch {torch.__version__} - Consider upgrading to 2.5.0")
        
        if torch.cuda.is_available():
            print(f"🎯 CUDA version: {torch.version.cuda}")
            print(f"💎 GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"🔥 GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   💾 Memory: {props.total_memory / 1e9:.1f} GB")
                print(f"   ⚙️  Compute capability: {props.major}.{props.minor}")
                
                # RTX 4050 specific checks
                if "RTX 4050" in torch.cuda.get_device_name(i):
                    print("   🎯 RTX 4050 detected - Optimized for this GPU!")
                    if props.major >= 8:  # Ada Lovelace architecture
                        print("   🚀 Ada Lovelace architecture - Full performance!")
        
        # Test performance improvements in PyTorch 2.5.0
        print("\n🧪 Testing PyTorch 2.5.0 performance improvements...")
        
        # Test with compilation if available
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            print("   🔥 Testing torch.compile (PyTorch 2.x optimization)...")
            
            def simple_model(x):
                return torch.nn.functional.relu(torch.mm(x, x.t()))
            
            # Compile the model for better performance
            compiled_model = torch.compile(simple_model)
            
            device = torch.device('cuda:0')
            x = torch.randn(500, 500, device=device)
            
            # Warm up
            _ = compiled_model(x)
            torch.cuda.synchronize()
            
            # Benchmark compiled vs non-compiled
            start = time.time()
            for _ in range(10):
                _ = simple_model(x)
            torch.cuda.synchronize()
            normal_time = time.time() - start
            
            start = time.time()  
            for _ in range(10):
                _ = compiled_model(x)
            torch.cuda.synchronize()
            compiled_time = time.time() - start
            
            print(f"   📊 Normal execution: {normal_time:.4f}s")
            print(f"   ⚡ Compiled execution: {compiled_time:.4f}s")
            print(f"   🚀 Speedup: {normal_time/compiled_time:.2f}x")
        
        # Test basic operations
        print("\n🧪 Testing basic tensor operations...")
        
        # CPU test
        x_cpu = torch.randn(1000, 1000)
        start_time = time.time()
        result_cpu = torch.mm(x_cpu, x_cpu.t())
        cpu_time = time.time() - start_time
        print(f"CPU matrix multiplication: {cpu_time:.4f}s")
        
        # GPU test if available
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            x_gpu = x_cpu.to(device)
            
            # Warm up
            _ = torch.mm(x_gpu, x_gpu.t())
            torch.cuda.synchronize()
            
            start_time = time.time()
            result_gpu = torch.mm(x_gpu, x_gpu.t())
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"GPU matrix multiplication: {gpu_time:.4f}s")
            print(f"Speedup: {cpu_time/gpu_time:.2f}x")
            
            # Memory test
            memory_used = torch.cuda.memory_allocated() / 1e6
            memory_cached = torch.cuda.memory_reserved() / 1e6
            print(f"GPU memory used: {memory_used:.1f} MB")
            print(f"GPU memory cached: {memory_cached:.1f} MB")
        
        print("✅ PyTorch 2.5.0 basics test passed!")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch basics test failed: {e}")
        traceback.print_exc()
        return False


def test_pytorch_geometric():
    """Test PyTorch Geometric functionality."""
    print("\n🔍 TESTING PYTORCH GEOMETRIC")
    print("=" * 30)
    
    try:
        import torch_geometric
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
        
        # Create simple graph
        edge_index = torch.tensor([[0, 1, 1, 2],
                                  [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 16)
        data = Data(x=x, edge_index=edge_index)
        
        print(f"Graph nodes: {data.num_nodes}")
        print(f"Graph edges: {data.num_edges}")
        print(f"Node features: {data.x.shape}")
        
        # Test GCN layer
        conv = GCNConv(16, 32)
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            data = data.to(device)
            conv = conv.to(device)
            print(f"Moved to GPU: {device}")
        
        # Forward pass
        out = conv(data.x, data.edge_index)
        print(f"GCN output shape: {out.shape}")
        print(f"Output device: {out.device}")
        
        print("✅ PyTorch Geometric test passed!")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch Geometric test failed: {e}")
        traceback.print_exc()
        return False


def test_document_gnn():
    """Test Document GNN model creation."""
    print("\n🔍 TESTING DOCUMENT GNN MODEL")
    print("=" * 30)
    
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        import torch.nn as nn
        
        # Create a simple DocumentGNN-like model
        class TestDocumentGNN(nn.Module):
            def __init__(self, num_features=20, hidden_dim=64, num_classes=6):
                super().__init__()
                self.embedding = nn.Linear(num_features, hidden_dim)
                self.conv1 = GCNConv(hidden_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.classifier = nn.Linear(hidden_dim, num_classes)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x, edge_index):
                x = torch.relu(self.embedding(x))
                x = torch.relu(self.conv1(x, edge_index))
                x = self.dropout(x)
                x = torch.relu(self.conv2(x, edge_index))
                x = self.dropout(x)
                return self.classifier(x)
        
        # Create model
        model = TestDocumentGNN()
        
        # Create test data (simulating document with 10 text blocks)
        num_nodes = 10
        x = torch.randn(num_nodes, 20)  # 20 features per node
        
        # Create edges (reading order + some spatial connections)
        edges = []
        # Reading order edges
        for i in range(num_nodes - 1):
            edges.extend([[i, i+1], [i+1, i]])  # Bidirectional
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Create labels
        y = torch.randint(0, 6, (num_nodes,))  # Random labels 0-5
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        print(f"Test graph - Nodes: {data.num_nodes}, Edges: {data.num_edges}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Move to GPU if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = data.to(device)
        
        print(f"Device: {device}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            output = model(data.x, data.edge_index)
            forward_time = time.time() - start_time
            
        print(f"Forward pass time: {forward_time*1000:.2f}ms")
        print(f"Output shape: {output.shape}")
        print(f"Predictions: {output.argmax(dim=1).tolist()}")
        
        # Test training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        train_time = time.time() - start_time
        
        print(f"Training step time: {train_time*1000:.2f}ms")
        print(f"Loss: {loss.item():.4f}")
        
        print("✅ Document GNN test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Document GNN test failed: {e}")
        traceback.print_exc()
        return False


def benchmark_performance():
    """Benchmark performance for different operations."""
    print("\n🔍 PERFORMANCE BENCHMARK")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping GPU benchmarks")
        return
    
    try:
        from torch_geometric.nn import GCNConv
        import torch.nn as nn
        
        device = torch.device('cuda:0')
        
        # Test different graph sizes
        sizes = [50, 100, 200, 500]
        
        print("Graph Size | Forward (ms) | Memory (MB)")
        print("-" * 40)
        
        for size in sizes:
            # Create graph
            x = torch.randn(size, 20, device=device)
            
            # Create edges (dense connectivity)
            edges = []
            for i in range(size):
                for j in range(min(5, size-1)):  # Connect to 5 neighbors
                    if i != (i + j + 1) % size:
                        edges.extend([[i, (i + j + 1) % size]])
            
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
            
            # Create simple model
            model = nn.Sequential(
                nn.Linear(20, 64),
                GCNConv(64, 64),
                nn.ReLU(),
                GCNConv(64, 6)
            ).to(device)
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model[1](model[0](x), edge_index)
            
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    h = model[0](x)  # Linear
                    h = model[1](h, edge_index)  # GCN
                    h = model[2](h)  # ReLU
                    out = model[3](h, edge_index)  # GCN
                
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times) * 1000
            memory_mb = torch.cuda.memory_allocated() / 1e6
            
            print(f"{size:>9} | {avg_time:>10.2f} | {memory_mb:>9.1f}")
            
            # Clear memory
            del model, x, edge_index
            torch.cuda.empty_cache()
        
        print("✅ Performance benchmark complete!")
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        traceback.print_exc()


def main():
    """Run all tests."""
    print("🚀 GPU AND CUDA TESTING FOR RTX 4050")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: PyTorch basics
    if test_pytorch_basics():
        tests_passed += 1
    
    # Test 2: PyTorch Geometric
    if test_pytorch_geometric():
        tests_passed += 1
    
    # Test 3: Document GNN
    if test_document_gnn():
        tests_passed += 1
    
    # Benchmark (optional)
    benchmark_performance()
    
    print("\n" + "=" * 50)
    print("🎉 TESTING COMPLETE")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Your setup is ready for GNN training.")
    else:
        print("❌ Some tests failed. Check the installation.")
        
    print("\nRecommended next steps:")
    print("1. If all tests passed: Run the graph construction pipeline")
    print("2. If tests failed: Check CUDA drivers and PyTorch installation")
    print("3. For memory issues: Reduce batch size or model dimensions")


if __name__ == "__main__":
    main()
