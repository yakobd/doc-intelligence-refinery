from src.agents.extractor import ExtractionRouter
import os

def test_full_pipeline():
    orchestrator = ExtractionRouter()
    raw_files = [f for f in os.listdir("data/raw") if f.endswith(".pdf")]
    
    print(f"\n🚀 Starting Refinery Pipeline for {len(raw_files)} files...\n")
    
    for file in raw_files:
        path = os.path.join("data/raw", file)
        print(f"📄 Processing: {file}...")
        try:
            result = orchestrator.process_document(path)
            print(f"✅ Success! Used: {result.strategy_used} | Pages: {len(result.pages)}\n")
        except Exception as e:
            print(f"❌ Failed {file}: {e}\n")

if __name__ == "__main__":
    test_full_pipeline()