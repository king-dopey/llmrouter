import sys
import logging

logging.basicConfig(level=logging.INFO)

def test_imports():
    """Verify all required packages are importable."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        sys.exit(1)

    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        sys.exit(1)

    try:
        import sentence_transformers
        print(f"✅ Sentence-Transformers {sentence_transformers.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Sentence-Transformers import failed: {e}")
        sys.exit(1)

def test_headroom_components():
    """Verify Headroom's components initialize correctly."""
    try:
        from headroom import compress
        print("✅ Headroom compress function imported successfully")
    except ImportError as e:
        print(f"❌ Headroom import failed: {e}")
        sys.exit(1)

    # Test code compressor initialization (Public API)
    try:
        from headroom.transforms import CodeAwareCompressor
        print("✅ CodeAwareCompressor class imported successfully")
    except ImportError as e:
        print(f"❌ CodeAwareCompressor import failed: {e}")
        sys.exit(1)

    # Test relevance compressor initialization (Direct Module Import)
    # RelevanceCompressor is not exported from headroom.transforms in v0.33.0
    # but exists in the module file. We import it directly to verify its existence.
    try:
        from headroom.transforms.relevance_compressor import RelevanceCompressor
        print("✅ RelevanceCompressor class imported successfully")
    except ImportError as e:
        print(f"❌ RelevanceCompressor import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
    test_headroom_components()