import sys
import os
# Ensure the src directory is in the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.triage_agent import TriageAgent

def run_triage_audit():
    agent = TriageAgent()
    data_dir = "data/raw"
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: {data_dir} directory not found.")
        return

    # Filter for PDFs in your raw data folder
    samples = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    print("\n" + "="*95)
    print(f"{'FILE NAME':<45} | {'ORIGIN':<15} | {'LAYOUT':<15} | {'DOMAIN'}")
    print("="*95)
    
    for filename in samples:
        path = os.path.join(data_dir, filename)
        try:
            profile = agent.profile_document(path)
            # Truncate long filenames for a clean table view
            display_name = (filename[:42] + '..') if len(filename) > 42 else filename
            
            print(f"{display_name:<45} | {profile.origin_type.value:<15} | {profile.layout_complexity.value:<15} | {profile.domain_hint}")
        except Exception as e:
            print(f"❌ Error profiling {filename}: {e}")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_triage_audit()