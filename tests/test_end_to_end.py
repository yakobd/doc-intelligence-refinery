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
            selected_strategy = result.metadata.get("selected_strategy", result.profile.selected_strategy.value)
            covered_pages = len({page for ldu in result.ldus for page in ldu.page_refs})
            print(f"✅ Success! Used: {selected_strategy} | Pages: {covered_pages}\n")
        except BaseException as e:
            print(f"❌ Failed {file}: {e}\n")

if __name__ == "__main__":
    test_full_pipeline()