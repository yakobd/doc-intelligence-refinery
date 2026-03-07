import logging
import json
from pathlib import Path
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

# Setup basic logging to see the Triage reasoning
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_validation(test_pdf_path: str):
    print("\n" + "="*50)
    print(f"MASTER-FIX VALIDATION: {Path(test_pdf_path).name}")
    print("="*50)

    # 1. Initialize Agents
    triage = TriageAgent()
    extractor = ExtractionRouter()

    # 2. Run Triage (Tests Heuristics #3 and #4)
    print(f"\n[1/3] Running Triage Analysis...")
    profile = triage.profile_document(test_pdf_path)
    
    print(f"  > Detected Origin: {profile.origin_type}")
    print(f"  > Layout Complexity: {profile.layout_complexity}")
    print(f"  > Selected Strategy: {profile.selected_strategy}")
    print(f"  > Estimated Chars: {profile.estimated_chars}")
    print(f"  > Estimated Cost: ${profile.estimated_cost:.6f}")

    # 3. Run Extraction (Tests Ledger #1 and Budget Guard #2)
    print(f"\n[2/3] Running Extraction Ledger & Budget Check...")
    result = extractor.process_document(test_pdf_path)

    # 4. Verify Ledger Output
    ledger_path = Path("logs/extraction_ledger.jsonl")
    if ledger_path.exists():
        with open(ledger_path, "r") as f:
            last_entry = json.loads(f.readlines()[-1])
        
        print(f"\n[3/3] Verifying Ledger Entry...")
        print(f"  > processing_time recorded: {'✅' if 'processing_time' in last_entry else '❌'}")
        print(f"  > cost_status: {last_entry.get('cost_status', 'N/A')}")
        print(f"  > final_cost: ${last_entry.get('final_cost', 0.0):.6f}")
    else:
        print("\n[!] Ledger file not found in logs/extraction_ledger.jsonl")

    print("\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    # Replace this with a path to one of your sample PDFs
    sample_path = "data/raw/Consumer Price Index May 2025.pdf" 
    if Path(sample_path).exists():
        run_validation(sample_path)
    else:
        print(f"Please update the 'sample_path' in the script to a valid PDF. Looking for: {sample_path}")